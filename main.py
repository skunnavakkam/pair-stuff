from transformer_lens import HookedTransformer
import torch

# Load the pre-trained model
model = HookedTransformer.from_pretrained("gpt2")
torch.set_printoptions(threshold=1_000_000)
torch.set_printoptions(linewidth=1_000_000)


# Ensure all parameters require gradients
for param in model.parameters():
    param.requires_grad = True

# Define the input text
input_text = "Peepee"

# Tokenize the input text
tokens = model.to_tokens(input_text)
print(tokens)

# Get regular embeddings
reg_embeddings = model.embed(tokens)

# Clone the first token's embedding to make it a leaf tensor
first_token_embedding = reg_embeddings[0, 0].clone().detach().requires_grad_(True)

# Define the optimizer for the first token's embedding
optimizer = torch.optim.Adam([first_token_embedding], lr=0.5)

# Optimization loop
num_steps = 200  # Number of optimization steps
for step in range(num_steps):
    # Zero the gradients
    optimizer.zero_grad()

    # Replace the first token's embedding in the reg_embeddings tensor
    reg_embeddings[0, 0] = first_token_embedding

    # Calculate the loss
    loss = model(reg_embeddings, tokens=tokens, return_type="loss", start_at_layer=0)

    # Backpropagate the loss with retain_graph=True
    loss.backward(retain_graph=True)

    # Optimize the first token's embedding
    optimizer.step()

    # Print the loss for every 10 steps
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item()}")

# Output the final optimized first token's embedding
print(first_token_embedding.shape)
optimized_embedding = first_token_embedding.detach()
print(optimized_embedding.shape)
print("Optimized first token's embedding:", optimized_embedding)

