from langchain_community.agent_toolkits import GmailToolkit

toolkit = GmailToolkit()
#in background it will look after the credentials.json file it should be in same directory
tools = toolkit.get_tools()
print(tools)

#custum authentication
from langchain_google_community.gmail.utils import (
    build_gmail_service,      # Changed from build_resource_service
    get_google_credentials    # Changed from get_gmail_credentials
)
from langchain_community.agent_toolkits import GmailToolkit

# 1. Use get_google_credentials
credentials = get_google_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)

# 2. Use build_gmail_service
api_resource = build_gmail_service(credentials=credentials)
# 3. Initialize toolkit
toolkit = GmailToolkit(api_resource=api_resource)


#now feeding the tools to agent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
llm = ChatGroq(model="openai/gpt-oss-120b",api_key=os.gentenv("GROQ_API_KEY"))

#create agent
from langchain.agents import create_agent

agent = create_agent(
    model = llm,
    tools=tools
)

#testing
example_query = "draft an email to test@test.com thanking them for the cofee"
events = agent.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    print(event["messages"][-1].pretty_print())

"""
=================OUTPUT========================

================================ Human Message =================================

draft an email to test@test.com thanking them for the cofee
None
================================== Ai Message ==================================
Tool Calls:
  create_gmail_draft (fc_97af106d-ee07-4292-b011-82c2467d6a9f)
 Call ID: fc_97af106d-ee07-4292-b011-82c2467d6a9f
  Args:
    message: Hi there,

I just wanted to send a quick note to thank you for the coffee you treated me to. It was a wonderful gesture and truly brightened my day. I appreciate your kindness and look forward to returning the favor sometime soon.

Thanks again!

Best regards,
[Your Name]
    subject: Thank You for the Coffee!
    to: ['test@test.com']
None
================================= Tool Message =================================
Name: create_gmail_draft

Draft created. Draft Id: r-3637598442258332111
None
================================== Ai Message ==================================

Your draft email to **test@test.com** has been created successfully.

**Subject:** Thank You for the Coffee!  
**To:** test@test.com  

**Message Preview:**

```
Hi there,

I just wanted to send a quick note to thank you for the coffee you treated me to. It was a wonderful gesture and truly brightened my day. I appreciate your kindness and look forward to returning the favor sometime soon.

Thanks again!

Best regards,
[Your Name]
```

Let me know if you’d like any edits or if you’d like to send it.
None
"""
