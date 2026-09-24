"""Small protocol and routing checks for the local proxy."""

import json
import unittest

import httpx

from serve_proxy import MODEL, create_app, route


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        words = sum(len(item["content"].split()) for item in messages)
        return {"input_ids": list(range(words + 3))}


class ProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_routing_and_protocol(self):
        seen = []

        def backend(request):
            if request.url.path in ("/health", "/healthcheck"):
                return httpx.Response(200, json={"status": "ok"})
            payload = json.loads(request.content)
            seen.append(payload)
            if payload["messages"][-1]["content"] == "backend error":
                return httpx.Response(503, json={"error": "backend unavailable"})
            if payload.get("stream"):
                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                      content=(b'data: {"choices":[{"delta":{"content":"A"}}]}\n\n'
                                               b'data: [DONE]\n\n'))
            return httpx.Response(200, json={"choices": [{"message": {"content": "A"}}]})

        upstream = httpx.AsyncClient(transport=httpx.MockTransport(backend))
        app = create_app("http://vllm", "http://lmcache", 10, Tokenizer(), upstream)
        first = [{"role": "user", "content": "Document one two three four five six seven eight. Who founded it?"}]
        followup = first + [
            {"role": "assistant", "content": "Alex."},
            {"role": "user", "content": "Which city is the headquarters in?"},
        ]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://proxy") as client:
            self.assertEqual((await client.get("/health")).status_code, 200)
            first_response = await client.post("/v1/chat/completions", json={
                "model": MODEL, "messages": first, "temperature": 0,
            })
            self.assertEqual(first_response.headers["x-jevkv-route"], "seed")
            response = await client.post("/v1/chat/completions", json={
                "model": MODEL, "messages": followup, "stream": True,
            })
            self.assertEqual(response.headers["x-jevkv-route"], "reuse")
            self.assertIn("data: [DONE]", response.text)
            self.assertEqual(seen[0]["cache_salt"], seen[1]["cache_salt"])
            self.assertEqual(seen[0]["temperature"], 0)
            exact = first + [
                {"role": "assistant", "content": "Alex."},
                {"role": "user", "content": "Quote the exact wording."},
            ]
            response = await client.post("/v1/chat/completions", json={
                "model": MODEL, "messages": exact,
            })
            self.assertEqual(response.headers["x-jevkv-route"], "fresh")
            self.assertNotEqual(seen[-1]["cache_salt"], seen[0]["cache_salt"])
            response = await client.post("/v1/chat/completions", json={
                "model": MODEL, "messages": first,
            }, headers={"X-JevKV-Cache": "bypass"})
            self.assertEqual(response.headers["x-jevkv-route"], "fresh")
            self.assertEqual((await client.post("/v1/chat/completions", json={
                "model": MODEL, "messages": first, "cache_salt": "unsafe",
            })).status_code, 400)
            response = await client.post("/v1/chat/completions", json={
                "model": MODEL, "messages": [{"role": "user", "content": "backend error"}],
            })
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"], "backend unavailable")
        await upstream.aclose()

    def test_short_prefix(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Which city?"},
        ]
        self.assertEqual(route({"model": MODEL, "messages": messages}, Tokenizer(), 10)[0],
                         "fresh")


if __name__ == "__main__":
    unittest.main()
