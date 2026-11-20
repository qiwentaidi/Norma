package tool

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestNewWebSearchConfig(t *testing.T) {
	if _, err := NewWebSearch(WebSearchConfig{Backend: "ddgs"}); err != nil {
		t.Fatalf("ddgs backend should build: %v", err)
	}
	if _, err := NewWebSearch(WebSearchConfig{}); err != nil {
		t.Fatalf("empty backend should default to ddgs: %v", err)
	}
	if _, err := NewWebSearch(WebSearchConfig{Backend: "brave-free"}); err == nil {
		t.Fatal("brave-free without key should error")
	}
	if _, err := NewWebSearch(WebSearchConfig{Backend: "brave-free", BraveAPIKey: "k"}); err != nil {
		t.Fatalf("brave-free with key should build: %v", err)
	}
	if _, err := NewWebSearch(WebSearchConfig{Backend: "nope"}); err == nil {
		t.Fatal("unknown backend should error")
	}
}

func TestDDGSSearchParse(t *testing.T) {
	// Minimal fixture mimicking the html.duckduckgo.com/html/ result markup,
	// including the //duckduckgo.com/l/?uddg= redirect wrapper.
	page := `<html><body>
	<div class="result">
	  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=x">First Result</a>
	  <a class="result__snippet">First snippet text.</a>
	</div>
	<div class="result">
	  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fb">Second Result</a>
	  <a class="result__snippet">Second snippet.</a>
	</div>
	</body></html>`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil || r.Form.Get("q") == "" {
			t.Errorf("expected form q param, got err=%v form=%v", err, r.Form)
		}
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte(page))
	}))
	defer srv.Close()

	p := &ddgsProvider{client: srv.Client(), endpoint: srv.URL}
	res, err := p.Search(context.Background(), "test query", 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res) != 2 {
		t.Fatalf("want 2 results, got %d: %+v", len(res), res)
	}
	if res[0].URL != "https://example.com/a" {
		t.Errorf("uddg redirect not unwrapped: %q", res[0].URL)
	}
	if res[0].Title != "First Result" || res[0].Description != "First snippet text." {
		t.Errorf("bad result[0]: %+v", res[0])
	}
	if res[0].Position != 1 || res[1].Position != 2 {
		t.Errorf("positions not 1-indexed: %d %d", res[0].Position, res[1].Position)
	}

	// limit is honored.
	res, _ = p.Search(context.Background(), "test", 1)
	if len(res) != 1 {
		t.Errorf("limit=1 should yield 1 result, got %d", len(res))
	}
}

func TestBraveSearchParse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-Subscription-Token"); got != "secret-key" {
			t.Errorf("missing/incorrect auth header: %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"web":{"results":[
			{"title":"T1","url":"https://a.example","description":"D1"},
			{"title":"T2","url":"https://b.example","description":"D2"}
		]}}`))
	}))
	defer srv.Close()

	p := &braveProvider{apiKey: "secret-key", client: srv.Client(), endpoint: srv.URL}
	res, err := p.Search(context.Background(), "q", 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res) != 2 || res[0].Title != "T1" || res[1].URL != "https://b.example" {
		t.Fatalf("bad parse: %+v", res)
	}
	if res[0].Position != 1 {
		t.Errorf("position not 1-indexed: %d", res[0].Position)
	}
}

func TestBraveHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()
	p := &braveProvider{apiKey: "k", client: srv.Client(), endpoint: srv.URL}
	if _, err := p.Search(context.Background(), "q", 5); err == nil {
		t.Fatal("expected error on HTTP 401")
	}
}

func TestUnwrapDDGHref(t *testing.T) {
	cases := map[string]string{
		"//duckduckgo.com/l/?uddg=https%3A%2F%2Fx.com%2Fy&rut=z": "https://x.com/y",
		"https://plain.example/path":                             "https://plain.example/path",
		"":                                                       "",
	}
	for in, want := range cases {
		if got := unwrapDDGHref(in); got != want {
			t.Errorf("unwrapDDGHref(%q) = %q, want %q", in, got, want)
		}
	}
	// Sanity: url encoding round-trips as expected.
	if _, err := url.Parse("//duckduckgo.com/l/?uddg=x"); err != nil {
		t.Fatal(err)
	}
}

func TestDeepSeekConfigValidation(t *testing.T) {
	full := WebSearchConfig{Backend: "deepseek", DeepSeekBaseURL: "https://api.deepseek.com/anthropic", DeepSeekAPIKey: "k", DeepSeekModel: "deepseek-chat"}
	if _, err := NewWebSearch(full); err != nil {
		t.Fatalf("fully configured deepseek backend should build: %v", err)
	}
	for _, tc := range []struct {
		name string
		mut  func(*WebSearchConfig)
	}{
		{"missing key", func(c *WebSearchConfig) { c.DeepSeekAPIKey = "" }},
		{"missing base url", func(c *WebSearchConfig) { c.DeepSeekBaseURL = "" }},
		{"missing model", func(c *WebSearchConfig) { c.DeepSeekModel = "" }},
	} {
		cfg := full
		tc.mut(&cfg)
		if _, err := NewWebSearch(cfg); err == nil {
			t.Fatalf("deepseek backend with %s should error", tc.name)
		}
	}
}

func TestDeepSeekMessagesURL(t *testing.T) {
	want := "https://api.deepseek.com/anthropic/v1/messages"
	for _, in := range []string{
		"https://api.deepseek.com/anthropic",
		"https://api.deepseek.com/anthropic/",
		"https://api.deepseek.com/anthropic/v1",
		"https://api.deepseek.com/anthropic/v1/messages",
	} {
		if got := deepseekMessagesURL(in); got != want {
			t.Errorf("deepseekMessagesURL(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestParseDeepSeekStream(t *testing.T) {
	// Mirrors a real DeepSeek stream: hits arrive whole inside content_block_start
	// (never as deltas), an empty title needs a host fallback, the same URL can
	// repeat across rounds, and a tool-side failure rides in the same array as
	// the hits. Unknown fields ("caller") must not break decoding.
	stream := `event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"call_00","name":"web_search","input":{},"caller":{"type":"direct"}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"x\"}"}}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"call_00","content":[{"type":"web_search_result","title":"First","url":"https://a.example/1","encrypted_content":"zzz"},{"type":"web_search_result","title":"","url":"https://b.example/2"},{"type":"web_search_result","title":"Dup","url":"https://a.example/1"}]}}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"web_search_tool_result","tool_use_id":"call_01","content":[{"type":"web_search_tool_result_error","error_code":"max_uses_exceeded"}]}}

event: content_block_start
data: {"type":"content_block_start","index":3,"content_block":{"type":"text","text":""}}

data: [DONE]

`
	got, err := parseDeepSeekStream(strings.NewReader(stream), 10)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("want 2 results (dup dropped, error item skipped), got %d: %+v", len(got), got)
	}
	if got[0].Title != "First" || got[0].URL != "https://a.example/1" || got[0].Position != 1 {
		t.Errorf("first result wrong: %+v", got[0])
	}
	// Empty title falls back to the host so the model still has a label.
	if got[1].Title != "b.example" {
		t.Errorf("empty title should fall back to host, got %q", got[1].Title)
	}
	// DeepSeek only returns encrypted page content, so there is never a snippet.
	if got[0].Description != "" {
		t.Errorf("description should stay empty, got %q", got[0].Description)
	}
}

func TestParseDeepSeekStreamLimitAndErrors(t *testing.T) {
	hits := `{"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","content":[{"type":"web_search_result","title":"A","url":"https://a.example"},{"type":"web_search_result","title":"B","url":"https://b.example"},{"type":"web_search_result","title":"C","url":"https://c.example"}]}}`
	got, err := parseDeepSeekStream(strings.NewReader("data: "+hits+"\n\n"), 2)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("limit should cap results at 2, got %d", len(got))
	}

	// A failure with no usable hits must surface rather than look like "no results".
	onlyErr := `data: {"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","content":[{"type":"web_search_tool_result_error","error_code":"max_uses_exceeded"}]}}`
	if _, err := parseDeepSeekStream(strings.NewReader(onlyErr+"\n\n"), 5); err == nil {
		t.Fatal("a result-less stream carrying an error_code should error")
	}

	// Stream-level error frames propagate too.
	streamErr := `data: {"type":"error","error":{"type":"overloaded_error","message":"boom"}}`
	if _, err := parseDeepSeekStream(strings.NewReader(streamErr+"\n\n"), 5); err == nil {
		t.Fatal("stream error frame should error")
	}

	// An empty but well-formed stream is "no results", not a failure.
	if res, err := parseDeepSeekStream(strings.NewReader("data: [DONE]\n\n"), 5); err != nil || len(res) != 0 {
		t.Fatalf("empty stream should yield no results and no error, got %v / %v", res, err)
	}
}

func TestDeepSeekSearchConfig(t *testing.T) {
	if _, err := NewWebSearch(WebSearchConfig{Backend: "deepseek-search"}); err == nil {
		t.Fatal("deepseek-search without a key should error")
	}
	ws, err := NewWebSearch(WebSearchConfig{Backend: "deepseek-search", DeepSeekAPIKey: "k"})
	if err != nil {
		t.Fatalf("deepseek-search with a key should build: %v", err)
	}
	if ws == nil {
		t.Fatal("nil tool")
	}
}

func TestDeepSeekSearchURL(t *testing.T) {
	const want = "https://api.deepseek.com/v1/tools/search"
	for _, in := range []string{"", "https://api.deepseek.com", "https://api.deepseek.com/", "https://api.deepseek.com/v1", "https://api.deepseek.com/v1/messages"} {
		got, err := deepseekSearchURL(in)
		if err != nil {
			t.Fatalf("%q: %v", in, err)
		}
		if got != want {
			t.Errorf("%q -> %q, want %q", in, got, want)
		}
	}
	// A gateway that keeps its own prefix must keep it.
	got, err := deepseekSearchURL("https://gw.example.com/deepseek/v1")
	if err != nil || got != "https://gw.example.com/deepseek/v1/tools/search" {
		t.Errorf("gateway URL: got %q err %v", got, err)
	}
	for _, bad := range []string{"ftp://api.deepseek.com", "api.deepseek.com", "not a url"} {
		if _, err := deepseekSearchURL(bad); err == nil {
			t.Errorf("%q should be rejected", bad)
		}
	}
}

func TestDeepSeekSearchParse(t *testing.T) {
	var got struct {
		Query string `json:"query"`
		Count int    `json:"count"`
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/tools/search" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if auth := r.Header.Get("Authorization"); auth != "Bearer secret" {
			t.Errorf("bad auth header: %q", auth)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"results":[
            {"title":"A","url":"https://a.example","summary":"long summary","snippet":"short"},
            {"title":"","url":"https://b.example","snippet":"only snippet"},
            {"title":"no url","url":"","summary":"dropped"},
            {"title":"D","url":"https://d.example","content":"body text"}
        ]}`))
	}))
	defer srv.Close()

	p := &deepseekSearchProvider{apiKey: "secret", client: srv.Client(), endpoint: srv.URL + "/v1/tools/search"}
	res, err := p.Search(context.Background(), "site:example.com 测试", 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	// The query must reach the engine verbatim, not wrapped in an instruction.
	if got.Query != "site:example.com 测试" || got.Count != 5 {
		t.Errorf("request body not literal: %+v", got)
	}
	if len(res) != 3 {
		t.Fatalf("want 3 usable results, got %d: %+v", len(res), res)
	}
	if res[0].Description != "long summary" {
		t.Errorf("summary should win over snippet: %+v", res[0])
	}
	if res[1].Description != "only snippet" || res[1].Title != "b.example" {
		t.Errorf("snippet/host fallbacks not applied: %+v", res[1])
	}
	if res[2].Description != "body text" {
		t.Errorf("content fallback not applied: %+v", res[2])
	}
	for n, r := range res {
		if r.Position != n+1 {
			t.Errorf("positions not 1-indexed: %+v", res)
		}
	}
}

func TestDeepSeekSearchLimitAndEmpty(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var got struct {
			Count int `json:"count"`
		}
		_ = json.NewDecoder(r.Body).Decode(&got)
		if got.Count != 2 {
			t.Errorf("limit not forwarded as count: %d", got.Count)
		}
		_, _ = w.Write([]byte(`{"results":[{"title":"A","url":"https://a.example"},{"title":"B","url":"https://b.example"},{"title":"C","url":"https://c.example"}]}`))
	}))
	defer srv.Close()

	p := &deepseekSearchProvider{apiKey: "k", client: srv.Client(), endpoint: srv.URL}
	res, err := p.Search(context.Background(), "q", 2)
	if err != nil || len(res) != 2 {
		t.Fatalf("limit not honored: n=%d err=%v", len(res), err)
	}

	empty := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"results":[]}`))
	}))
	defer empty.Close()
	p = &deepseekSearchProvider{apiKey: "k", client: empty.Client(), endpoint: empty.URL}
	res, err = p.Search(context.Background(), "no hits", 5)
	if err != nil {
		t.Fatalf("empty results must be a success, got %v", err)
	}
	if len(res) != 0 {
		t.Fatalf("want no results, got %+v", res)
	}
}

func TestDeepSeekSearchErrors(t *testing.T) {
	for _, tc := range []struct {
		status int
		want   string
	}{
		{http.StatusUnauthorized, "API key"},
		{http.StatusPaymentRequired, "balance"},
		{http.StatusBadRequest, "rejected the request"},
		{http.StatusTooManyRequests, "rate-limiting"},
		{http.StatusInternalServerError, "unavailable"},
	} {
		tc := tc
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(tc.status)
			_, _ = w.Write([]byte(`{"error":{"message":"details"}}`))
		}))
		p := &deepseekSearchProvider{apiKey: "k", client: srv.Client(), endpoint: srv.URL}
		_, err := p.Search(context.Background(), "q", 1)
		srv.Close()
		if err == nil || !strings.Contains(err.Error(), tc.want) || !strings.Contains(err.Error(), "details") {
			t.Errorf("HTTP %d: got %v, want message containing %q and the API detail", tc.status, err, tc.want)
		}
	}
}

func TestDeepSeekSearchRejectsBadQuery(t *testing.T) {
	// No server: an invalid query must fail before any request is issued.
	p := &deepseekSearchProvider{apiKey: "k", client: http.DefaultClient, endpoint: "http://127.0.0.1:1/v1/tools/search"}
	for _, q := range []string{"", strings.Repeat("字", 129)} {
		if _, err := p.Search(context.Background(), q, 5); err == nil || !strings.Contains(err.Error(), "1-128") {
			t.Errorf("query %q: want length error, got %v", q, err)
		}
	}
}
