from flask import Flask, render_template, request
import openai

app = Flask(__name__)

# Set your OpenAI API key securely (you should use environment variables in production)
openai.api_key = "YOUR_API_KEY"

# Predefined prompts based on customer queries
PROMPTS = {
    "reset_password": (
        "Respond to the user in a friendly and professional tone. Provide clear, step-by-step instructions on how "
        "to reset their account password. Assume the user is trying to access their account on a website and may need "
        "to verify their identity. Mention where to find the 'Forgot Password' link, and what to expect after clicking it. "
        "Keep the instructions concise and easy to follow."
    ),
    "order_status": (
        "Reply in a polite and helpful tone. Ask the user for any missing information needed to check their order status "
        "(such as order number or email address). Structure your response in a short paragraph followed by bullet points "
        "outlining the steps they can take to view their order status online or contact customer support. Mention typical "
        "processing and delivery times if applicable."
    ),
    "return_policy": (
        "Provide a clear and informative response in a warm, reassuring tone. Use a FAQ-style format to summarize the key "
        "points of the return policy, such as time limits, condition of returned items, return shipping, and refund processing. "
        "Avoid legal jargon. Include a link or direction to where they can read the full return policy if possible."
    )
}

@app.route('/', methods=['GET', 'POST'])
def index():
    response_text = ""
    if request.method == 'POST':
        topic = request.form.get('topic')
        prompt = PROMPTS.get(topic, "Please respond to the user in a helpful and friendly tone.")
        
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful and friendly customer support assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        response_text = completion['choices'][0]['message']['content']
    
    return render_template('index.html', response=response_text)

if __name__ == '__main__':
    app.run(debug=True)
