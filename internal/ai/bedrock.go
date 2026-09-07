package ai

// AWS Bedrock Mantle: OpenAI GPT models through the Responses API and Claude
// through the Messages API, on AWS's endpoints, with AWS credentials.
//
// No AWS SDK. Bedrock authenticates with a bearer token that is nothing more
// than a SigV4-presigned URL of `https://bedrock.amazonaws.com/?Action=
// CallWithBearerToken`, base64-encoded — about a hundred lines of HMAC below.
// Credentials come from the three variables every AWS tool understands
// (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN); a profile, an
// SSO login, or an assumed role becomes those with one command:
//
//	eval "$(aws configure export-credentials --profile prod --format env)"
//
// That keeps pgbot's promise: keys only from the environment, no config files
// read, no calls to STS or instance metadata, and one static binary.

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	// A minted token lives this long at most (the service allows up to 12h;
	// pgbot makes one call and has no reason to hold a longer-lived secret).
	bedrockTokenTTL = 15 * time.Minute
	bedrockSTSHost  = "bedrock.amazonaws.com"
)

// awsCredentials are the environment credentials a token is minted from.
type awsCredentials struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string    // empty for long-lived keys
	Expires         time.Time // zero when not known
}

// awsCredentialsFromEnv reads the standard variables. AWS_CREDENTIAL_EXPIRATION
// is what `aws configure export-credentials --format env` emits next to them;
// honoring it keeps a minted token from outliving the credentials behind it.
func awsCredentialsFromEnv() (awsCredentials, error) {
	c := awsCredentials{
		AccessKeyID:     envOr("AWS_ACCESS_KEY_ID", ""),
		SecretAccessKey: envOr("AWS_SECRET_ACCESS_KEY", ""),
		SessionToken:    envOr("AWS_SESSION_TOKEN", ""),
	}
	if c.AccessKeyID == "" || c.SecretAccessKey == "" {
		return c, errors.New("no AWS credentials for Bedrock — set AWS_BEARER_TOKEN_BEDROCK (a Bedrock API key), " +
			"or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (and AWS_SESSION_TOKEN) and pgbot mints a short-lived token; " +
			"a profile or SSO login exports them with: eval \"$(aws configure export-credentials --format env)\"")
	}
	if exp := envOr("AWS_CREDENTIAL_EXPIRATION", ""); exp != "" {
		t, err := time.Parse(time.RFC3339, exp)
		if err != nil {
			return c, fmt.Errorf("AWS_CREDENTIAL_EXPIRATION %q is not an RFC 3339 timestamp", exp)
		}
		c.Expires = t
	}
	return c, nil
}

func bedrockModel(model, base, key string, httpc *http.Client) (LanguageModel, error) {
	region := firstEnv("AWS_REGION", "AWS_DEFAULT_REGION")
	if region == "" {
		region = "us-east-1"
	}
	if model == "" {
		model = "openai." + defaultOpenAIModel
	}
	anthropic := strings.HasPrefix(model, "anthropic.")
	if base == "" {
		base = "https://bedrock-mantle." + region + ".api.aws"
		if anthropic {
			base += "/anthropic"
		} else {
			base += "/openai/v1"
		}
	}
	base = trimURL(base)
	// Never forward a supplied or minted bearer token through a redirect.
	httpc.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	if key == "" {
		creds, err := awsCredentialsFromEnv()
		if err != nil {
			return nil, err
		}
		u, err := url.Parse(base)
		if err != nil || u.Scheme != "https" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.Host != "bedrock-mantle."+region+".api.aws" {
			return nil, fmt.Errorf("AWS credentials are only sent to the Bedrock Mantle HTTPS endpoint for region %s; set AWS_REGION to the endpoint's region", region)
		}
		httpc.Transport = &bedrockAuth{creds: creds, region: region, host: u.Host, anthropic: anthropic, next: http.DefaultTransport}
	}
	if anthropic {
		p := &AnthropicProvider{APIKey: key, BaseURL: base, HTTP: httpc, Label: "bedrock"}
		return p.LanguageModel(context.Background(), model)
	}
	p := &ResponsesProvider{APIKey: key, BaseURL: base, HTTP: httpc, Label: "bedrock", ReasoningEffort: envOr("PGBOT_AI_REASONING_EFFORT", "")}
	return p.LanguageModel(context.Background(), model)
}

// bedrockAuth mints a fresh token per request. Minting is local HMAC work, so
// there is nothing to cache or refresh; the token's TTL is bounded by the
// credentials' own expiry.
type bedrockAuth struct {
	creds        awsCredentials
	region, host string
	anthropic    bool // Messages API takes x-api-key; the Responses API a Bearer
	next         http.RoundTripper
}

func (a *bedrockAuth) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.URL.Scheme != "https" || req.URL.Host != a.host {
		return nil, errors.New("refusing to send AWS credentials outside the configured Mantle endpoint")
	}
	token, err := bedrockToken(a.creds, a.region, time.Now().UTC())
	if err != nil {
		return nil, err
	}
	clone := req.Clone(req.Context())
	if a.anthropic {
		clone.Header.Set("x-api-key", token)
	} else {
		clone.Header.Set("Authorization", "Bearer "+token)
	}
	return a.next.RoundTrip(clone)
}

// bedrockToken builds the bearer token AWS's own token generators produce: a
// SigV4 query-presigned POST to bedrock.amazonaws.com?Action=CallWithBearerToken
// (empty-payload hash; UNSIGNED-PAYLOAD yields an invalid token), with the
// scheme stripped and "&Version=1" appended, base64-encoded, prefixed.
func bedrockToken(creds awsCredentials, region string, now time.Time) (string, error) {
	ttl := bedrockTokenTTL
	if !creds.Expires.IsZero() && creds.Expires.Sub(now) < ttl {
		ttl = creds.Expires.Sub(now)
	}
	if ttl < time.Second {
		return "", errors.New("AWS credentials have expired; renew your AWS login and export them again")
	}
	amzDate := now.UTC().Format("20060102T150405Z")
	scope := amzDate[:8] + "/" + region + "/bedrock/aws4_request"
	params := map[string]string{
		"Action":              "CallWithBearerToken",
		"X-Amz-Algorithm":     "AWS4-HMAC-SHA256",
		"X-Amz-Credential":    creds.AccessKeyID + "/" + scope,
		"X-Amz-Date":          amzDate,
		"X-Amz-Expires":       strconv.FormatInt(int64(ttl/time.Second), 10),
		"X-Amz-SignedHeaders": "host",
	}
	if creds.SessionToken != "" {
		params["X-Amz-Security-Token"] = creds.SessionToken
	}
	query := sigv4Query(params)
	emptyPayload := sha256.Sum256(nil)
	canonical := strings.Join([]string{
		http.MethodPost, "/", query,
		"host:" + bedrockSTSHost, "", // canonical headers, then the blank line
		"host", hex.EncodeToString(emptyPayload[:]),
	}, "\n")
	canonicalHash := sha256.Sum256([]byte(canonical))
	toSign := strings.Join([]string{"AWS4-HMAC-SHA256", amzDate, scope, hex.EncodeToString(canonicalHash[:])}, "\n")
	key := []byte("AWS4" + creds.SecretAccessKey)
	for _, part := range []string{amzDate[:8], region, "bedrock", "aws4_request"} {
		key = hmacSHA256(key, part)
	}
	signature := hex.EncodeToString(hmacSHA256(key, toSign))
	presigned := bedrockSTSHost + "/?" + query + "&X-Amz-Signature=" + signature
	return "bedrock-api-key-" + base64.StdEncoding.EncodeToString([]byte(presigned+"&Version=1")), nil
}

// sigv4Query renders params as SigV4's canonical query string: keys sorted,
// every key and value RFC 3986-encoded (only unreserved characters bare, hex
// upper-case, space as %20 — not the form encoding net/url produces).
func sigv4Query(params map[string]string) string {
	keys := make([]string, 0, len(params))
	for k := range params {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, sigv4Escape(k)+"="+sigv4Escape(params[k]))
	}
	return strings.Join(parts, "&")
}

func sigv4Escape(s string) string {
	const hexDigits = "0123456789ABCDEF"
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case 'A' <= c && c <= 'Z', 'a' <= c && c <= 'z', '0' <= c && c <= '9', c == '-', c == '_', c == '.', c == '~':
			b.WriteByte(c)
		default:
			b.WriteByte('%')
			b.WriteByte(hexDigits[c>>4])
			b.WriteByte(hexDigits[c&15])
		}
	}
	return b.String()
}

func hmacSHA256(key []byte, data string) []byte {
	h := hmac.New(sha256.New, key)
	h.Write([]byte(data))
	return h.Sum(nil)
}
