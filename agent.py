# GitLab-Jira Integration Agent using Google ADK
# Step-by-step implementation with setup requirements

"""
SETUP REQUIREMENTS:

1. GitLab Setup:
   - Go to GitLab → User Settings → Access Tokens
   - Create a Personal Access Token with scopes:
     * api (full API access)
     * read_user
     * read_repository
   - Note down the token

2. Jira Setup:
   - Go to Jira → Profile → Personal Access Tokens (or Account Settings → Security → API tokens)
   - Create an API token
   - Note down your Jira email and token
   - Note your Jira base URL (e.g., https://yourcompany.atlassian.net)

3. Google AI Studio Setup:
   - Go to https://aistudio.google.com/
   - Create an API key
   - Note down the API key

4. Environment Setup:
   pip install google-adk requests python-dotenv

5. Create .env file:
   GITLAB_TOKEN=your_gitlab_token
   GITLAB_BASE_URL=https://gitlab.com/api/v4  # or your GitLab instance URL
   JIRA_EMAIL=your_jira_email
   JIRA_TOKEN=your_jira_token
   JIRA_BASE_URL=https://yourcompany.atlassian.net
   GOOGLE_API_KEY=your_google_api_key
"""

import os
import requests
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()

# Google ADK imports
from google.adk.agents import Agent
from google.adk.tools import BaseTool
import google.generativeai as genai

@dataclass
class GitLabConfig:
    base_url: str
    token: str

@dataclass
class JiraConfig:
    base_url: str
    email: str
    token: str

class GitLabService:
    def __init__(self, config: GitLabConfig):
        self.config = config
        self.headers = {
            'Authorization': f'Bearer {config.token}',
            'Content-Type': 'application/json'
        }
    
    def fetch_merge_request(self, project_id: str, mr_iid: str) -> Dict[str, Any]:
        """Fetch merge request details"""
        url = f"{self.config.base_url}/projects/{project_id}/merge_requests/{mr_iid}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_merge_request_diff(self, project_id: str, mr_iid: str) -> List[Dict[str, Any]]:
        """Get merge request diff using Merge Request Diff API"""
        url = f"{self.config.base_url}/projects/{project_id}/merge_requests/{mr_iid}/diffs"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_merge_request_changes(self, project_id: str, mr_iid: str) -> Dict[str, Any]:
        """Get detailed changes in merge request"""
        url = f"{self.config.base_url}/projects/{project_id}/merge_requests/{mr_iid}/changes"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def update_merge_request_status(self, project_id: str, mr_iid: str, action: str) -> Dict[str, Any]:
        """Update merge request status (merge, close, reopen)"""
        if action == "merge":
            url = f"{self.config.base_url}/projects/{project_id}/merge_requests/{mr_iid}/merge"
            response = requests.put(url, headers=self.headers)
        elif action == "close":
            url = f"{self.config.base_url}/projects/{project_id}/merge_requests/{mr_iid}"
            data = {"state_event": "close"}
            response = requests.put(url, headers=self.headers, json=data)
        elif action == "reopen":
            url = f"{self.config.base_url}/projects/{project_id}/merge_requests/{mr_iid}"
            data = {"state_event": "reopen"}
            response = requests.put(url, headers=self.headers, json=data)
        else:
            raise ValueError(f"Invalid action: {action}")
        
        response.raise_for_status()
        return response.json()
    
    def approve_merge_request(self, project_id: str, mr_iid: str) -> Dict[str, Any]:
        """Approve merge request"""
        url = f"{self.config.base_url}/projects/{project_id}/merge_requests/{mr_iid}/approve"
        response = requests.post(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def unapprove_merge_request(self, project_id: str, mr_iid: str) -> Dict[str, Any]:
        """Unapprove merge request"""
        url = f"{self.config.base_url}/projects/{project_id}/merge_requests/{mr_iid}/unapprove"
        response = requests.post(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

class JiraService:
    def __init__(self, config: JiraConfig):
        self.config = config
        # Create basic auth header
        auth_string = f"{config.email}:{config.token}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        self.headers = {
            'Authorization': f'Basic {encoded_auth}',
            'Content-Type': 'application/json'
        }
    
    def fetch_issue(self, issue_key: str) -> Dict[str, Any]:
        """Fetch Jira issue details"""
        url = f"{self.config.base_url}/rest/api/3/issue/{issue_key}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def update_issue_status(self, issue_key: str, transition_id: str) -> Dict[str, Any]:
        """Update Jira issue status using transition"""
        url = f"{self.config.base_url}/rest/api/3/issue/{issue_key}/transitions"
        data = {
            "transition": {
                "id": transition_id
            }
        }
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        return {"success": True, "message": f"Issue {issue_key} status updated"}
    
    def get_issue_transitions(self, issue_key: str) -> List[Dict[str, Any]]:
        """Get available transitions for an issue"""
        url = f"{self.config.base_url}/rest/api/3/issue/{issue_key}/transitions"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()["transitions"]
    
    def add_comment(self, issue_key: str, comment: str) -> Dict[str, Any]:
        """Add comment to Jira issue"""
        url = f"{self.config.base_url}/rest/api/3/issue/{issue_key}/comment"
        data = {
            "body": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": comment
                            }
                        ]
                    }
                ]
            }
        }
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()

# Custom Tool Classes
class FetchGitLabMRTool(BaseTool):
    def __init__(self, gitlab_service):
        super().__init__()
        self.gitlab_service = gitlab_service
        self.name = "fetch_gitlab_mr"
        self.description = "Fetch GitLab merge request details"
    
    def run(self, project_id: str, mr_iid: str) -> str:
        """Fetch GitLab merge request details"""
        try:
            mr_data = self.gitlab_service.fetch_merge_request(project_id, mr_iid)
            return f"Merge Request #{mr_iid} fetched successfully:\n" + \
                   f"Title: {mr_data['title']}\n" + \
                   f"State: {mr_data['state']}\n" + \
                   f"Author: {mr_data['author']['name']}\n" + \
                   f"Source Branch: {mr_data['source_branch']}\n" + \
                   f"Target Branch: {mr_data['target_branch']}\n" + \
                   f"Description: {mr_data.get('description', 'No description')}"
        except Exception as e:
            return f"Error fetching merge request: {str(e)}"

class GetMRDiffTool(BaseTool):
    def __init__(self, gitlab_service):
        super().__init__()
        self.gitlab_service = gitlab_service
        self.name = "get_mr_diff"
        self.description = "Get merge request diff for code comparison"
    
    def run(self, project_id: str, mr_iid: str) -> str:
        """Get merge request diff for code comparison"""
        try:
            diff_data = self.gitlab_service.get_merge_request_diff(project_id, mr_iid)
            changes_data = self.gitlab_service.get_merge_request_changes(project_id, mr_iid)
            
            summary = f"Merge Request #{mr_iid} Changes:\n"
            summary += f"Files changed: {len(changes_data.get('changes', []))}\n\n"
            
            for change in changes_data.get('changes', [])[:5]:  # Limit to first 5 files
                summary += f"File: {change['new_path']}\n"
                summary += f"  - Additions: +{change.get('additions', 0)}\n"
                summary += f"  - Deletions: -{change.get('deletions', 0)}\n"
                if change.get('diff'):
                    # Show first few lines of diff
                    diff_lines = change['diff'].split('\n')[:10]
                    summary += f"  - Preview:\n    " + "\n    ".join(diff_lines) + "\n\n"
            
            return summary
        except Exception as e:
            return f"Error fetching merge request diff: {str(e)}"

class UpdateMRStatusTool(BaseTool):
    def __init__(self, gitlab_service):
        super().__init__()
        self.gitlab_service = gitlab_service
        self.name = "update_mr_status"
        self.description = "Update merge request status (approve/unapprove/merge/close/reopen)"
    
    def run(self, project_id: str, mr_iid: str, action: str) -> str:
        """Update merge request status (approve/unapprove/merge/close/reopen)"""
        try:
            if action in ["approve", "unapprove"]:
                if action == "approve":
                    result = self.gitlab_service.approve_merge_request(project_id, mr_iid)
                else:
                    result = self.gitlab_service.unapprove_merge_request(project_id, mr_iid)
            else:
                result = self.gitlab_service.update_merge_request_status(project_id, mr_iid, action)
            
            return f"Merge request #{mr_iid} {action} successful"
        except Exception as e:
            return f"Error updating merge request status: {str(e)}"

class FetchJiraTicketTool(BaseTool):
    def __init__(self, jira_service):
        super().__init__()
        self.jira_service = jira_service
        self.name = "fetch_jira_ticket"
        self.description = "Fetch Jira ticket details"
    
    def run(self, issue_key: str) -> str:
        """Fetch Jira ticket details"""
        try:
            issue_data = self.jira_service.fetch_issue(issue_key)
            fields = issue_data['fields']
            
            return f"Jira Ticket {issue_key} fetched successfully:\n" + \
                   f"Summary: {fields['summary']}\n" + \
                   f"Status: {fields['status']['name']}\n" + \
                   f"Priority: {fields.get('priority', {}).get('name', 'Not set')}\n" + \
                   f"Assignee: {fields.get('assignee', {}).get('displayName', 'Unassigned')}\n" + \
                   f"Description: {fields.get('description', 'No description')}"
        except Exception as e:
            return f"Error fetching Jira ticket: {str(e)}"

class UpdateJiraStatusTool(BaseTool):
    def __init__(self, jira_service):
        super().__init__()
        self.jira_service = jira_service
        self.name = "update_jira_status"
        self.description = "Update Jira ticket status"
    
    def run(self, issue_key: str, status_name: str) -> str:
        """Update Jira ticket status"""
        try:
            # Get available transitions
            transitions = self.jira_service.get_issue_transitions(issue_key)
            
            # Find the transition ID for the desired status
            transition_id = None
            for transition in transitions:
                if transition['to']['name'].lower() == status_name.lower():
                    transition_id = transition['id']
                    break
            
            if transition_id:
                result = self.jira_service.update_issue_status(issue_key, transition_id)
                return f"Jira ticket {issue_key} status updated to {status_name}"
            else:
                available_statuses = [t['to']['name'] for t in transitions]
                return f"Status '{status_name}' not available. Available statuses: {', '.join(available_statuses)}"
        except Exception as e:
            return f"Error updating Jira ticket status: {str(e)}"

class AddJiraCommentTool(BaseTool):
    def __init__(self, jira_service):
        super().__init__()
        self.jira_service = jira_service
        self.name = "add_jira_comment"
        self.description = "Add comment to Jira ticket"
    
    def run(self, issue_key: str, comment: str) -> str:
        """Add comment to Jira ticket"""
        try:
            result = self.jira_service.add_comment(issue_key, comment)
            return f"Comment added to Jira ticket {issue_key} successfully"
        except Exception as e:
            return f"Error adding comment to Jira ticket: {str(e)}"

class GitLabJiraAgent:
    def __init__(self):
        # Initialize configurations
        self.gitlab_config = GitLabConfig(
            base_url=os.getenv('GITLAB_BASE_URL'),
            token=os.getenv('GITLAB_TOKEN')
        )
        
        self.jira_config = JiraConfig(
            base_url=os.getenv('JIRA_BASE_URL'),
            email=os.getenv('JIRA_EMAIL'),
            token=os.getenv('JIRA_TOKEN')
        )
        
        # Initialize services
        self.gitlab_service = GitLabService(self.gitlab_config)
        self.jira_service = JiraService(self.jira_config)
        
        # Initialize Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Initialize tools
        self.tools = [
            FetchGitLabMRTool(self.gitlab_service),
            GetMRDiffTool(self.gitlab_service),
            UpdateMRStatusTool(self.gitlab_service),
            FetchJiraTicketTool(self.jira_service),
            UpdateJiraStatusTool(self.jira_service),
            AddJiraCommentTool(self.jira_service)
        ]
        
        # Initialize ADK Agent
        self.agent = Agent(
            name="GitLab-Jira Integration Agent",
            instructions="""
            You are a GitLab-Jira integration agent that helps manage merge requests and Jira tickets.
            You can:
            1. Fetch GitLab merge requests and their diffs
            2. Compare code changes in merge requests
            3. Update merge request status (approve/unapprove/merge/close)
            4. Fetch Jira tickets
            5. Update Jira ticket status
            6. Add comments to Jira tickets
            
            Always provide clear status updates and handle errors gracefully.
            """,
            model=self.model,
            tools=self.tools
        )
    
    def execute_workflow(self, project_id: str, mr_iid: str, jira_issue_key: str):
        """Execute the complete workflow"""
        workflow_prompt = f"""
        Execute the following workflow:
        1. Fetch GitLab merge request {mr_iid} from project {project_id}
        2. Get the code diff and analyze changes
        3. Fetch Jira ticket {jira_issue_key}
        4. Based on the merge request status and code quality, decide if we should:
           - Approve the merge request
           - Update the Jira ticket status appropriately
           - Add a comment to the Jira ticket with merge request details
        
        Please execute this workflow step by step and provide status updates.
        """
        
        return self.agent.run(workflow_prompt)
    
    def chat(self, message: str):
        """Chat interface for the agent"""
        return self.agent.run(message)

# Example usage and testing
def main():
    """Main function to demonstrate the agent"""
    
    # Initialize the agent
    agent = GitLabJiraAgent()
    
    print("GitLab-Jira Integration Agent initialized successfully!")
    print("\nAvailable commands:")
    print("1. Execute workflow: agent.execute_workflow(project_id, mr_iid, jira_issue_key)")
    print("2. Chat with agent: agent.chat('your message')")
    print("\nExample usage:")
    print("response = agent.execute_workflow('123', '45', 'PROJ-789')")
    print("response = agent.chat('Fetch merge request 45 from project 123 and show me the diff')")
    
    # Interactive mode
    while True:
        try:
            user_input = input("\nEnter your command (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break
            
            response = agent.chat(user_input)
            print(f"\nAgent Response: {response}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

"""
USAGE EXAMPLES:

1. Basic workflow execution:
   agent = GitLabJiraAgent()
   response = agent.execute_workflow("123", "45", "PROJ-789")

2. Individual operations:
   agent.chat("Fetch merge request 45 from project 123")
   agent.chat("Get diff for merge request 45 in project 123")
   agent.chat("Approve merge request 45 in project 123")
   agent.chat("Fetch Jira ticket PROJ-789")
   agent.chat("Update Jira ticket PROJ-789 status to 'In Progress'")

3. Complex workflows:
   agent.chat("Fetch MR 45 from project 123, analyze the diff, then update PROJ-789 accordingly")

CONFIGURATION NOTES:
- Make sure your GitLab token has appropriate permissions
- Jira transitions vary by project configuration
- Test with a small project first
- Monitor API rate limits
"""