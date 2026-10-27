
import os
import json
import time

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError
from pydantic import BaseModel, Field, ValidationError


# Load environment variables from .env
load_dotenv()

# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. "
        "Please add it to your .env file."
    )

# Initialize Gemini client
client = genai.Client(api_key=api_key)


# Define the expected JSON structure
class ResumeData(BaseModel):
    name: str = Field(
        description="Candidate's full name"
    )
    email: str = Field(
        description="Candidate's email address"
    )
    skills: list[str] = Field(
        description="Candidate's skills"
    )
    experience: str = Field(
        description="Work experience or fresher status"
    )


# Extract and validate resume information
def extract_resume_data(text: str) -> dict:

    prompt = f"""
    Extract the candidate's information from the resume below.

    Return a JSON object with these fields:
    name, email, skills, experience.

    Do not invent missing information.
    Use an empty string for missing text fields
    and an empty list for missing skills.

    Resume:
    {text}
    """

    # Retry temporary server errors up to 3 times
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config={
                    "response_mime_type": "application/json"
                },
            )

            if not response.text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            # Convert JSON text into a dictionary
            data = json.loads(response.text)

            # Validate extracted data
            validated_data = ResumeData.model_validate(data)

            return validated_data.model_dump()

        except ServerError:
            if attempt == 2:
                raise

            wait_seconds = 2 ** (attempt + 1)

            print(
                f"Gemini server unavailable. "
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        "Unable to get a response after multiple attempts."
    )


# Main program
if __name__ == "__main__":

    sample_resume = """
    Name: Priya Kumar
    Email: priya@example.com
    Skills: Python, SQL, Pandas
    Experience: Fresher
    """

    try:
        # Extract resume data
        result = extract_resume_data(sample_resume)

        print("\nValidated JSON Output:")
        print(json.dumps(result, indent=4))

        # Save output to a JSON file
        with open(
            "output.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(result, file, indent=4)

        print("\nSuccess! Output saved to output.json")

    except (ValidationError, json.JSONDecodeError) as error:
        print("\nJSON or validation error:", error)

    except ServerError as error:
        print(
            "\nGemini server is still unavailable after retries."
        )
        print("Please wait and try again.")
        print("Details:", error)

    except Exception as error:
        print("\nAn error occurred:", error)
