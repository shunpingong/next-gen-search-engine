import textgrad as tg
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

tg.set_backward_engine("experimental:openrouter/openai/gpt-4o", override=True, cache=False)

model = tg.BlackboxLLM(tg.get_engine("experimental:openrouter/openai/gpt-3.5-turbo", cache=False))
question_string = ("If it takes 1 hour to dry 25 shirts under the sun, "
                   "how long will it take to dry 30 shirts under the sun? "
                   "Reason step by step")

question = tg.Variable(question_string,
                       role_description="question to the LLM",
                       requires_grad=False)

answer = model(question)

print(answer)

# Initialize the system prompt
system_prompt = tg. Variable ("You are a helpful language model . Think step by step .",
                                requires_grad = True ,
                                role_description =" system prompt to the language model ")

# Set up the model object ’ parameterized by ’ the prompt .
model = tg. BlackboxLLM ( system_prompt = system_prompt )

# Optimize the system prompt
optimizer = tg.TextualGradientDescent ( parameters =[ system_prompt ])

for iteration in range ( max_iterations ):
    batch_x , batch_y = next ( train_loader )
    optimizer . zero_grad ()
    # Do the forward pass
    responses = model ( batch_x )
    losses =[loss_fn ( response , y ) for ( response , y) in zip ( responses , batch_y ) ]
    total_loss = tg.sum( losses )
    # Perform the backward pass and compute gradients
    total_loss.backward()
    # Update the system prompt
    optimizer.step()
