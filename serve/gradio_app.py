"""Gradio web interface for the LLM."""

import gradio as gr


def create_gradio_app(model, params, tokenizer, cfg):
    """Build and return a Gradio Blocks app."""
    from src.generate import generate

    def chat(message, history, temperature, top_k, max_tokens):
        context = ""
        for msg in history:
            context += f"{msg['content']}\n"
        context += message

        output = generate(
            model=model,
            params=params,
            tokenizer=tokenizer,
            prompt=context,
            max_new_tokens=int(max_tokens),
            temperature=temperature,
            top_k=int(top_k),
            max_seq_len=cfg["max_seq_len"],
        )
        response = output[len(context):].strip()
        return response

    def plain_generate(prompt, temperature, top_k, max_tokens):
        output = generate(
            model=model,
            params=params,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=int(max_tokens),
            temperature=temperature,
            top_k=int(top_k),
            max_seq_len=cfg["max_seq_len"],
        )
        return output

    with gr.Blocks(title="LLM-JAX") as app:
        gr.Markdown("# 🧠 LLM-JAX\nA GPT-style model built from scratch with JAX/Flax")

        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(placeholder="Type a message...", label="Your message", lines=2)

            with gr.Row():
                temp_chat = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                topk_chat = gr.Slider(1, 100, value=50, step=1, label="Top-k")
                max_tok_chat = gr.Slider(16, 512, value=128, step=16, label="Max tokens")

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.ClearButton([msg, chatbot], value="Clear")

            def respond(message, chat_history, temperature, top_k, max_tokens):
                bot_reply = chat(message, chat_history, temperature, top_k, max_tokens)
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": bot_reply})
                return "", chat_history

            send_btn.click(respond, [msg, chatbot, temp_chat, topk_chat, max_tok_chat], [msg, chatbot])
            msg.submit(respond, [msg, chatbot, temp_chat, topk_chat, max_tok_chat], [msg, chatbot])

        with gr.Tab("📝 Text Completion"):
            prompt_input = gr.Textbox(placeholder="Enter a prompt...", label="Prompt", lines=4)
            output_box = gr.Textbox(label="Generated text", lines=10, interactive=False)

            with gr.Row():
                temp_gen = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                topk_gen = gr.Slider(1, 100, value=50, step=1, label="Top-k")
                max_tok_gen = gr.Slider(16, 512, value=256, step=16, label="Max tokens")

            gen_btn = gr.Button("Generate", variant="primary")
            gen_btn.click(plain_generate, [prompt_input, temp_gen, topk_gen, max_tok_gen], output_box)

        with gr.Tab("ℹ️ Model Info"):
            import jax
            n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
            gr.Markdown(f"""
### Model Configuration
| Setting | Value |
|---------|-------|
| Parameters | **{n_params / 1e6:.1f}M** |
| Layers | {cfg['n_layers']} |
| Dimensions | {cfg['d_model']} |
| Heads | {cfg['n_heads']} |
| Context length | {cfg['max_seq_len']} |
| Vocab size | {cfg['vocab_size']} |
| Device | {jax.devices()[0]} |
""")

    return app