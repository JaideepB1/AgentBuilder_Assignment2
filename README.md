# HireGraph

HireGraph is a LangGraph-based hiring workflow which evaluates a candidate's resume against a job description and gives a final hiring recommendation.

## Setup

Install dependencies:

```
pip install -r requirements.txt
```

Create a `.env` file and add your OpenAI API key.

Run the application:

```bash
python main.py
```

## Usage

The project includes strong, borderline and weak candidates who are evaluated against a backend engineering role description.

Running the application evaluates each candidate and displays their score, seniority level, recommendation and workflow status.