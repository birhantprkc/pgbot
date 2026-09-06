package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
)

func tokenQuery(t *testing.T, token string) url.Values {
	t.Helper()
	if !strings.HasPrefix(token, "bedrock-api-key-") {
		t.Fatal("missing token prefix")
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
	creds := aws.Credentials{AccessKeyID: "AKIDEXAMPLE", SecretAccessKey: "dummy-secret", SessionToken: "session/+= token"}
	token, err := bedrockToken(context.Background(), creds, "us-east-1", now)
	if err != nil {
		t.Fatal(err)
	}
	q := tokenQuery(t, token)
	// Golden signature from AWS's Python aws-bedrock-token-generator with these
	// dummy credentials, frozen timestamp, region, and 900-second expiry.
	if q.Get("X-Amz-Signature") != "c51d43f3459d73b462fc95dca8da87f70d1a65920d0bfd32f1d6761e47485a2f" {
		t.Fatal("signature differs from AWS reference generator")
	}
	if q.Get("Version") != "1" || q.Get("X-Amz-Expires") != "900" || q.Get("X-Amz-Security-Token") != creds.SessionToken {
		t.Fatal("incorrect token envelope")
	}
	creds.CanExpire = true
	creds.Expires = now.Add(90 * time.Second)
	token, err = bedrockToken(context.Background(), creds, "us-east-1", now)
	if err != nil {
		t.Fatal(err)
	}
	if tokenQuery(t, token).Get("X-Amz-Expires") != "90" {
		t.Fatal("token must not outlive credentials")
	}
	creds.Expires = now
	if _, err := bedrockToken(context.Background(), creds, "us-east-1", now); err == nil {
		t.Fatal("expired credentials accepted")
	}
	creds.CanExpire = false
	creds.SessionToken = ""
	token, err = bedrockToken(context.Background(), creds, "us-east-1", now)
	if err != nil {
		t.Fatal(err)
	}
	if _, exists := tokenQuery(t, token)["X-Amz-Security-Token"]; exists {
		t.Fatal("static credentials must omit session token")
	}
}

type bedrockTestTransport func(*http.Request) (*http.Response, error)

func (f bedrockTestTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func isolateAWS(t *testing.T) {
	t.Helper()
	clearEnv(t)
	for _, k := range []string{"AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI"} {
		t.Setenv(k, "")
	}
	t.Setenv("AWS_EC2_METADATA_DISABLED", "true")
	t.Setenv("AWS_CONFIG_FILE", filepath.Join(t.TempDir(), "config"))
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", filepath.Join(t.TempDir(), "credentials"))
}

func TestBedrockIAMProfile(t *testing.T) {
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
			t.Setenv("AWS_PROFILE", "test-profile")
			if err := os.WriteFile(os.Getenv("AWS_CONFIG_FILE"), []byte("[profile test-profile]\nregion = us-west-2\naws_access_key_id = AKIDEXAMPLE\naws_secret_access_key = dummy-secret\naws_session_token = session/+= token\n"), 0600); err != nil {
				t.Fatal(err)
			}
			m, err := Resolve()
			if err != nil {
				t.Fatal(err)
			}
			var client *http.Client
			header := "Authorization"
			switch m := m.(type) {
			case *responsesModel:
				client = m.provider.HTTP
			case *anthropicModel:
				client = m.provider.HTTP
				header = "x-api-key"
			default:
				t.Fatalf("unexpected model type %T", m)
			}
			if m.Provider() != "bedrock" || !strings.Contains(m.Endpoint(), "us-west-2") {
				t.Fatal("profile region or provider label lost")
			}
			auth := client.Transport.(*bedrockAuth)
			calls := 0
			auth.next = bedrockTestTransport(func(r *http.Request) (*http.Response, error) {
				calls++
				if r.URL.Path != tc.path {
					t.Errorf("incorrect API path: %s", r.URL.Path)
				}
				q := tokenQuery(t, strings.TrimPrefix(r.Header.Get(header), "Bearer "))
				if !strings.Contains(q.Get("X-Amz-Credential"), "/us-west-2/bedrock/") {
					t.Error("incorrect signing region")
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
	t.Setenv("AWS_PROFILE", "nonexistent")
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-override")
	t.Setenv("PGBOT_AI_API_KEY", "explicit-override")
	m, err := Resolve()
	if err != nil {
		t.Fatal(err)
	}
	p := m.(*responsesModel).provider
	if p.APIKey != "explicit-override" || p.HTTP.Transport != nil {
		t.Fatal("explicit token must bypass IAM")
	}
	t.Setenv("PGBOT_AI_API_KEY", "")
	m, err = Resolve()
	if err != nil {
		t.Fatal(err)
	}
	if m.(*responsesModel).provider.APIKey != "bedrock-override" {
		t.Fatal("Bedrock token override lost")
	}
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "")
	t.Setenv("AWS_PROFILE", "")
	t.Setenv("OPENAI_API_KEY", "unrelated-key")
	m, err = Resolve()
	if err != nil {
		t.Fatal(err)
	}
	if _, err = m.Generate(context.Background(), Call{Prompt: "hello"}); err == nil || !strings.Contains(err.Error(), "resolve AWS credentials") {
		t.Fatalf("expected missing IAM credentials error, got %v", err)
	}
	t.Setenv("PGBOT_AI_BASE_URL", "https://example.com/openai/v1")
	if _, err := Resolve(); err == nil {
		t.Fatal("IAM authentication must reject non-Mantle endpoints")
	}
	t.Setenv("AWS_REGION", "us-east-1")
	t.Setenv("PGBOT_AI_BASE_URL", "https://bedrock-mantle.us-west-2.api.aws/openai/v1")
	if _, err := Resolve(); err == nil {
		t.Fatal("IAM region mismatch accepted")
	}
}
