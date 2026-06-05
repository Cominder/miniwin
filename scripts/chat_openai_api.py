"""调用兼容 OpenAI 的本地服务（如 scripts/serve_openai_api.py），支持流式 reasoning_content。"""
import argparse

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(
        description="交互式 OpenAI 兼容客户端（需先启动 serve_openai_api 等服务）"
    )
    parser.add_argument("--base_url", default="http://127.0.0.1:8998/v1", help="API base URL")
    parser.add_argument("--model", default="miniwin", help="模型名（需与服务端一致）")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--stream", type=int, default=1, choices=[0, 1])
    parser.add_argument("--open_thinking", type=int, default=0, choices=[0, 1])
    parser.add_argument("--history_messages_num", type=int, default=2, help="携带最近几条消息（0=仅当前轮）")
    args = parser.parse_args()

    client = OpenAI(api_key="ollama", base_url=args.base_url)
    stream = bool(args.stream)
    conversation_history = []

    while True:
        try:
            query = input("[Q]: ")
        except EOFError:
            print()
            break
        if not query.strip():
            continue
        conversation_history.append({"role": "user", "content": query})
        extra = {}
        if args.open_thinking:
            extra["extra_body"] = {"chat_template_kwargs": {"open_thinking": True}}
        msgs = (
            conversation_history[-args.history_messages_num :]
            if args.history_messages_num
            else conversation_history[-1:]
        )
        response = client.chat.completions.create(
            model=args.model,
            messages=msgs,
            stream=stream,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            **extra,
        )
        if not stream:
            msg = response.choices[0].message
            assistant_res = msg.content or ""
            rc = getattr(msg, "reasoning_content", None)
            if rc:
                print("[思考]: ", rc)
            print("[A]: ", assistant_res)
        else:
            print("[A]: ", end="", flush=True)
            assistant_res = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                r = getattr(delta, "reasoning_content", None) or ""
                c = delta.content or ""
                if r:
                    print(f"\033[90m{r}\033[0m", end="", flush=True)
                if c:
                    print(c, end="", flush=True)
                assistant_res += c

        conversation_history.append({"role": "assistant", "content": assistant_res})
        print("\n\n")


if __name__ == "__main__":
    main()
