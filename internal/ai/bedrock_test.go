package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"
)

func tokenQuery(t *testing.T, token string) url.Values {
	t.Helper()
	if !strings.HasPrefix(token, "bedrock-api-key-") {
		t.Fatalf("missing token prefix in %q", token)
	}
	data, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(token, "bedrock-api-key-"))
	if err != nil {
		t.Fatal(err)
	}
	u, err := url.Parse("https://" + string(data))
	if err != nil {
		t.Fatal(err)
	}
	if u.Host != "bedrock.amazonaws.com" || u.Path != "/" {
		t.Fatal("incorrect signing target")
	}
	return u.Query()
}

func TestBedrockToken(t *testing.T) {
	now := time.Date(2026, 9, 5, 12, 0, 0, 0, time.UTC)
	creds := awsCredentials{AccessKeyID: "AKIDEXAMPLE", SecretAccessKey: "dummy-secret", SessionToken: "session/+= token"}
	token, err := bedrockToken(creds, "us-east-1", now)
	if err != nil {
		t.Fatal(err)
	}
	q := tokenQuery(t, token)
	// Golden signature from AWS's Python aws-bedrock-token-generator with these
	// dummy credentials, frozen timestamp, region, and 900-second expiry — the
	// proof that the hand-rolled presigner matches the SDK it replaced.
	if q.Get("X-Amz-Signature") != "c51d43f3459d73b462fc95dca8da87f70d1a65920d0bfd32f1d6761e47485a2f" {
		t.Fatalf("signature differs from AWS reference generator: %s", q.Get("X-Amz-Signature"))
	}
	if q.Get("Version") != "1" || q.Get("X-Amz-Expires") != "900" || q.Get("X-Amz-Security-Token") != creds.SessionToken {
		t.Fatal("incorrect token envelope")
	}
	if q.Get("X-Amz-Credential") != "AKIDEXAMPLE/20260905/us-east-1/bedrock/aws4_request" || q.Get("X-Amz-SignedHeaders") != "host" {
		t.Fatal("incorrect credential scope")
	}

	creds.Expires = now.Add(90 * time.Second)
	token, err = bedrockToken(creds, "us-east-1", now)
	if err != nil {
		t.Fatal(err)
	}
	if tokenQuery(t, token).Get("X-Amz-Expires") != "90" {
		t.Fatal("token must not outlive credentials")
	}
	creds.Expires = now
	if _, err := bedrockToken(creds, "us-east-1", now); err == nil {
		t.Fatal("expired credentials accepted")
	}
	creds.Expires = time.Time{}
	creds.SessionToken = ""
	token, err = bedrockToken(creds, "us-east-1", now)
	if err != nil {
		t.Fatal(err)
	}
	if _, exists := tokenQuery(t, token)["X-Amz-Security-Token"]; exists {
		t.Fatal("static credentials must omit session token")
	}
}

// The canonical query must use RFC 3986 escaping, not net/url's form encoding:
// a space is %20 and '+' is %2B, or the signature does not verify.
func TestSigv4Escape(t *testing.T) {
	if got := sigv4Escape("session/+= token~-_."); got != "session%2F%2B%3D%20token~-_." {
		t.Fatalf("sigv4Escape = %q", got)
	}
	if got := sigv4Query(map[string]string{"b": "2", "A": "1", "X-Amz-Date": "x"}); got != "A=1&X-Amz-Date=x&b=2" {
		t.Fatalf("sigv4Query ordering = %q", got)
	}
}

type bedrockTestTransport func(*http.Request) (*http.Response, error)

func (f bedrockTestTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func isolateAWS(t *testing.T) {
	t.Helper()
	clearEnv(t)
	for _, k := range []string{"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_CREDENTIAL_EXPIRATION",
		"AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE"} {
		t.Setenv(k, "")
	}
}

func TestBedrockEnvCredentials(t *testing.T) {
	for _, tc := range []struct {
		model, base, path string
	}{
		{"openai.gpt-5.6-terra", "", "/openai/v1/responses"},
		{"anthropic.claude-sonnet-5", "", "/anthropic/v1/messages"},
		{"anthropic.claude-sonnet-5", "https://bedrock-mantle.us-west-2.api.aws", "/v1/messages"},
		{"openai.gpt-5.6-terra", "https://bedrock-mantle.us-west-2.api.aws/anthropic", "/anthropic/responses"},
	} {
		t.Run(tc.model+tc.path, func(t *testing.T) {
			model := tc.model
			isolateAWS(t)
			t.Setenv("PGBOT_AI_PROVIDER", "bedrock")
			t.Setenv("PGBOT_AI_MODEL", model)
			t.Setenv("PGBOT_AI_BASE_URL", tc.base)
			t.Setenv("AWS_REGION", "us-west-2")
			t.Setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
			t.Setenv("AWS_SECRET_ACCESS_KEY", "dummy-secret")
			t.Setenv("AWS_SESSION_TOKEN", "session/+= token")
			m, err := Resolve()
			if err != nil {
				t.Fatal(err)
			}
			var client *http.Client
			header, other := "Authorization", "x-api-key"
			switch m := m.(type) {
			case *responsesModel:
				client = m.provider.HTTP
			case *anthropicModel:
				client = m.provider.HTTP
				header, other = "x-api-key", "Authorization"
			default:
				t.Fatalf("unexpected model type %T", m)
			}
			if m.Provider() != "bedrock" || !strings.Contains(m.Endpoint(), "us-west-2") {
				t.Fatal("region or provider label lost")
			}
			auth := client.Transport.(*bedrockAuth)
			calls := 0
			auth.next = bedrockTestTransport(func(r *http.Request) (*http.Response, error) {
				calls++
				if r.URL.Path != tc.path {
					t.Errorf("incorrect API path: %s", r.URL.Path)
				}
				// The header follows the model family, never the URL path.
				q := tokenQuery(t, strings.TrimPrefix(r.Header.Get(header), "Bearer "))
				if v := r.Header.Get(other); strings.Contains(v, "bedrock-api-key-") {
					t.Errorf("token also sent in %s", other)
				}
				if !strings.Contains(q.Get("X-Amz-Credential"), "/us-west-2/bedrock/") || q.Get("X-Amz-Security-Token") != "session/+= token" {
					t.Error("incorrect signing region or session token")
				}
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				if body["model"] != model {
					t.Error("model override lost")
				}
				if header == "x-api-key" && r.Header.Get("anthropic-version") != anthropicVersion {
					t.Error("missing Anthropic version")
				}
				return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"status":"completed","stop_reason":"end_turn","content":[{"type":"text","text":"OK"}],"output":[{"type":"message","content":[{"type":"output_text","text":"OK"}]}]}`))}, nil
			})
			out, err := m.Generate(context.Background(), Call{Prompt: "hello"})
			if err != nil {
				t.Fatal(err)
			}
			if out.Text != "OK" || calls != 1 {
				t.Fatal("generation failed")
			}
			if client.CheckRedirect == nil || client.CheckRedirect(nil, nil) != http.ErrUseLastResponse {
				t.Fatal("credentialed redirects must be disabled")
			}
			r, _ := http.NewRequest("POST", "https://example.com/responses", nil)
			if _, err := auth.RoundTrip(r); err == nil || calls != 1 {
				t.Fatal("credentials sent outside Mantle")
			}
		})
	}
}

func TestBedrockAuthConfiguration(t *testing.T) {
	isolateAWS(t)
	t.Setenv("PGBOT_AI_PROVIDER", "bedrock")
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-override")
	t.Setenv("PGBOT_AI_API_KEY", "explicit-override")
	m, err := Resolve()
	if err != nil {
		t.Fatal(err)
	}
	p := m.(*responsesModel).provider
	if p.APIKey != "explicit-override" || p.HTTP.Transport != nil {
		t.Fatal("an explicit token must be used as-is, with no minting transport")
	}
	if m.Model() != "openai."+defaultOpenAIModel || m.Endpoint() != "https://bedrock-mantle.us-east-1.api.aws/openai/v1" {
		t.Fatalf("defaults: model=%s endpoint=%s", m.Model(), m.Endpoint())
	}
	t.Setenv("PGBOT_AI_API_KEY", "")
	m, err = Resolve()
	if err != nil {
		t.Fatal(err)
	}
	if m.(*responsesModel).provider.APIKey != "bedrock-override" {
		t.Fatal("Bedrock token override lost")
	}

	// No token and no access keys: say exactly what to set, and never touch a
	// profile or the AWS config files.
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "")
	t.Setenv("AWS_PROFILE", "some-profile")
	t.Setenv("OPENAI_API_KEY", "unrelated-key")
	_, err = Resolve()
	if err == nil || !strings.Contains(err.Error(), "aws configure export-credentials") {
		t.Fatalf("expected a missing-credentials error naming the export command, got %v", err)
	}

	t.Setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "dummy-secret")
	t.Setenv("PGBOT_AI_BASE_URL", "https://example.com/openai/v1")
	if _, err := Resolve(); err == nil {
		t.Fatal("access keys must not be sent to a non-Mantle endpoint")
	}
	t.Setenv("AWS_REGION", "us-east-1")
	t.Setenv("PGBOT_AI_BASE_URL", "https://bedrock-mantle.us-west-2.api.aws/openai/v1")
	if _, err := Resolve(); err == nil {
		t.Fatal("region mismatch accepted")
	}

	// AWS_CREDENTIAL_EXPIRATION (emitted by `aws configure export-credentials`)
	// bounds the minted token; an unparseable value is refused rather than ignored.
	t.Setenv("PGBOT_AI_BASE_URL", "")
	t.Setenv("AWS_CREDENTIAL_EXPIRATION", "not-a-time")
	if _, err := Resolve(); err == nil {
		t.Fatal("malformed AWS_CREDENTIAL_EXPIRATION accepted")
	}
	t.Setenv("AWS_CREDENTIAL_EXPIRATION", time.Now().UTC().Add(-time.Minute).Format(time.RFC3339))
	m, err = Resolve()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Generate(context.Background(), Call{Prompt: "hello"}); err == nil || !strings.Contains(err.Error(), "expired") {
		t.Fatalf("expired credentials should fail at token minting, got %v", err)
	}
}
