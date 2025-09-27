from openai import OpenAI

# initialize client
client = OpenAI()

# Example support ticket text
ticket = "I cannot log into my account even though my password is correct."

# Define categories we want
categories = ["Billing", "Technical Issue", "Account Access", "Product Feedback"]

# Build a zero-shot prompt
prompt = f"""
You are a system that classifies customer support tickets.
Ticket: "{ticket}"
Categories: {", ".join(categories)}
Assign the top 3 most relevant categories in order of likelihood.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)

print("Predicted Tags:", response.choices[0].message.content)
