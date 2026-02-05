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