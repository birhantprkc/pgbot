package ai

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	"github.com/aws/aws-sdk-go-v2/config"
)

func bedrockModel(model, base, key string, httpc *http.Client) (LanguageModel, error) {
	region := firstEnv("AWS_REGION", "AWS_DEFAULT_REGION")
	var cfg aws.Config
	if key == "" {
		var err error
		cfg, err = config.LoadDefaultConfig(context.Background())
		if err != nil {
			return nil, fmt.Errorf("load AWS configuration: %w", err)
		}
		if region == "" {
			region = cfg.Region
		}
	}
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
	// Never forward a supplied or IAM-derived bearer token through a redirect.
	httpc.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	if key == "" {
		u, err := url.Parse(base)
		if err != nil || u.Scheme != "https" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.Host != "bedrock-mantle."+region+".api.aws" {
			return nil, fmt.Errorf("IAM authentication requires a Bedrock Mantle HTTPS endpoint matching AWS region %s; set AWS_REGION to the endpoint region", region)
		}
		httpc.Transport = &bedrockAuth{credentials: cfg.Credentials, region: region, host: u.Host, anthropic: anthropic, next: http.DefaultTransport}
	}
	if anthropic {
		p := &AnthropicProvider{APIKey: key, BaseURL: base, HTTP: httpc, Label: "bedrock"}
		return p.LanguageModel(context.Background(), model)
	}
	p := &ResponsesProvider{APIKey: key, BaseURL: base, HTTP: httpc, Label: "bedrock", ReasoningEffort: envOr("PGBOT_AI_REASONING_EFFORT", "")}
	return p.LanguageModel(context.Background(), model)
}

// The SDK caches and refreshes credentials. Minting per request is local signing,
// so no separate token cache or refresh goroutine is needed.
type bedrockAuth struct {
	credentials  aws.CredentialsProvider
	region, host string
	anthropic    bool
	next         http.RoundTripper
}

func (a *bedrockAuth) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.URL.Scheme != "https" || req.URL.Host != a.host {
		return nil, fmt.Errorf("refusing to send AWS credentials outside the configured Mantle endpoint")
	}
	creds, err := a.credentials.Retrieve(req.Context())
	if err != nil {
		return nil, fmt.Errorf("resolve AWS credentials (AWS_PROFILE or the default credential chain): %w", err)
	}
	token, err := bedrockToken(req.Context(), creds, a.region, time.Now().UTC())
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

func bedrockToken(ctx context.Context, creds aws.Credentials, region string, now time.Time) (string, error) {
	ttl := 15 * time.Minute
	if creds.CanExpire && creds.Expires.Sub(now) < ttl {
		ttl = creds.Expires.Sub(now)
	}
	if ttl < time.Second {
		return "", fmt.Errorf("AWS credentials have expired; renew your AWS login")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://bedrock.amazonaws.com/?Action=CallWithBearerToken", nil)
	if err != nil {
		return "", err
	}
	query := req.URL.Query()
	query.Set("X-Amz-Expires", strconv.FormatInt(int64(ttl/time.Second), 10))
	req.URL.RawQuery = query.Encode()
	// This request is only presigned, never sent. The empty payload hash matters:
	// UNSIGNED-PAYLOAD yields a different signature and an invalid bearer token.
	signed, _, err := v4.NewSigner().PresignHTTP(ctx, creds, req, fmt.Sprintf("%x", sha256.Sum256(nil)), "bedrock", region, now)
	if err != nil {
		return "", fmt.Errorf("sign Bedrock token: %w", err)
	}
	return "bedrock-api-key-" + base64.StdEncoding.EncodeToString([]byte(strings.TrimPrefix(signed, "https://")+"&Version=1")), nil
}
