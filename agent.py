"""
GitLab PR & Jira Integration Agent using Google ADK

This agent:
1. Fetches PR requests from GitLab
2. Checks if PRs are approved
3. Merges approved PRs
4. Updates corresponding Jira tickets

Requirements:
pip install google-adk requests python-dotenv
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import requests
from dotenv import load_dotenv
load_dotenv()

# Google ADK imports
from google.adk.agents import LlmAgent
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools import FunctionTool

# Load environment variables
load_dotenv()

# Google AI Studio API Key
GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

# GitLab Configuration
GITLAB_BASE_URL=os.getenv('GITLAB_BASE_URL')
GITLAB_TOKEN=os.getenv('GITLAB_TOKEN')
GITLAB_PROJECT_ID=os.getenv('GITLAB_PROJECT_ID')

# Jira Configuration
JIRA_BASE_URL=os.getenv('JIRA_BASE_URL')
JIRA_EMAIL=os.getenv('JIRA_EMAIL')
JIRA_API_TOKEN=os.getenv('JIRA_API_TOKEN')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PRInfo:
    """Data structure for PR information"""
    id: int
    title: str
    description: str
    state: str
    merge_status: str
    source_branch: str
    target_branch: str
    author: str
    web_url: str
    approvals: List[str]
    jira_ticket: Optional[str] = None

@dataclass
class JiraTicket:
    """Data structure for Jira ticket information"""
    key: str
    summary: str
    status: str
    assignee: str

class GitLabToolkit(BaseToolset):
    """Toolkit for GitLab operations"""
    
    def __init__(self, base_url: str, token: str, project_id: str):
        super().__init__()
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.project_id = project_id
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    def get_tools(self):
        """Return the tools provided by this toolset"""
        return [
            FunctionTool(self.fetch_merge_requests),
            FunctionTool(self.get_merge_request_approvals),
            FunctionTool(self.merge_request),
            FunctionTool(self.extract_jira_ticket_from_branch),
        ]
    
    def close(self):
        """Clean up resources (no resources to clean up for this toolkit)"""
        pass
    
    def fetch_merge_requests(self, state: str = "opened") -> List[Dict]:
        """
        Fetch merge requests from GitLab
        
        Args:
            state: The state of merge requests to fetch (opened, closed, merged)
            
        Returns:
            List of merge request dictionaries
        """
        try:
            url = f"{self.base_url}/api/v4/projects/{self.project_id}/merge_requests"
            params = {'state': state, 'per_page': 100}
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            mrs = response.json()
            logger.info(f"Fetched {len(mrs)} merge requests")
            return mrs
            
        except requests.RequestException as e:
            logger.error(f"Error fetching merge requests: {e}")
            return []
    
    def get_merge_request_approvals(self, mr_iid: int) -> Dict:
        """
        Get approvals for a specific merge request
        
        Args:
            mr_iid: The IID of the merge request
            
        Returns:
            Dictionary containing approval information
        """
        try:
            url = f"{self.base_url}/api/v4/projects/{self.project_id}/merge_requests/{mr_iid}/approvals"
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            approvals = response.json()
            logger.info(f"Fetched approvals for MR {mr_iid}")
            return approvals
            
        except requests.RequestException as e:
            logger.error(f"Error fetching approvals for MR {mr_iid}: {e}")
            return {}
    
    def merge_request(self, mr_iid: int, commit_message: str = None) -> Dict:
        """
        Merge a specific merge request
        
        Args:
            mr_iid: The IID of the merge request to merge
            commit_message: Optional commit message for the merge
            
        Returns:
            Dictionary containing merge result
        """
        try:
            url = f"{self.base_url}/api/v4/projects/{self.project_id}/merge_requests/{mr_iid}/merge"
            
            data = {}
            if commit_message:
                data['merge_commit_message'] = commit_message
            
            response = requests.put(url, headers=self.headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Successfully merged MR {mr_iid}")
            return result
            
        except requests.RequestException as e:
            logger.error(f"Error merging MR {mr_iid}: {e}")
            return {"error": str(e)}
    
    def extract_jira_ticket_from_branch(self, branch_name: str) -> Optional[str]:
        """
        Extract Jira ticket key from branch name
        
        Args:
            branch_name: The name of the branch
            
        Returns:
            Jira ticket key if found, None otherwise
        """
        import re
        
        # Common patterns for Jira tickets in branch names
        patterns = [
            r'([A-Z]{2,}-\d+)',  # Standard format: ABC-123
            r'([A-Z]{2,}_\d+)',  # Underscore format: ABC_123
        ]
        
        for pattern in patterns:
            match = re.search(pattern, branch_name.upper())
            if match:
                return match.group(1).replace('_', '-')
        
        return None

class JiraToolkit(BaseToolset):
    """Toolkit for Jira operations"""
    
    def __init__(self, base_url: str, username: str, api_token: str):
        super().__init__()
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.api_token = api_token
        self.auth = (username, api_token)
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    def get_tools(self):
        """Return the tools provided by this toolset"""
        return [
            FunctionTool(self.get_ticket_info),
            FunctionTool(self.update_ticket_status),
            FunctionTool(self.add_comment),
        ]
    
    def close(self):
        """Clean up resources (no resources to clean up for this toolkit)"""
        pass
    
    def get_ticket_info(self, ticket_key: str) -> Dict:
        """
        Get information about a Jira ticket
        
        Args:
            ticket_key: The key of the Jira ticket (e.g., PROJ-123)
            
        Returns:
            Dictionary containing ticket information
        """
        try:
            url = f"{self.base_url}/rest/api/3/issue/{ticket_key}"
            
            response = requests.get(url, auth=self.auth, headers=self.headers)
            response.raise_for_status()
            
            ticket = response.json()
            logger.info(f"Fetched info for ticket {ticket_key}")
            return ticket
            
        except requests.RequestException as e:
            logger.error(f"Error fetching ticket {ticket_key}: {e}")
            return {}
    
    def update_ticket_status(self, ticket_key: str, status: str, comment: str = None) -> Dict:
        """
        Update the status of a Jira ticket
        
        Args:
            ticket_key: The key of the Jira ticket
            status: The new status for the ticket
            comment: Optional comment to add
            
        Returns:
            Dictionary containing update result
        """
        try:
            # Get available transitions
            transitions_url = f"{self.base_url}/rest/api/3/issue/{ticket_key}/transitions"
            transitions_response = requests.get(transitions_url, auth=self.auth, headers=self.headers)
            transitions_response.raise_for_status()
            
            transitions = transitions_response.json()['transitions']
            target_transition = None
            
            for transition in transitions:
                if transition['to']['name'].lower() == status.lower():
                    target_transition = transition
                    break
            
            if not target_transition:
                logger.error(f"Status '{status}' not available for ticket {ticket_key}")
                return {"error": f"Status '{status}' not available"}
            
            # Execute transition
            transition_data = {
                "transition": {"id": target_transition['id']}
            }
            
            if comment:
                transition_data["update"] = {
                    "comment": [{"add": {"body": comment}}]
                }
            
            response = requests.post(transitions_url, auth=self.auth, 
                                   headers=self.headers, json=transition_data)
            response.raise_for_status()
            
            logger.info(f"Updated ticket {ticket_key} to status '{status}'")
            return {"success": True, "status": status}
            
        except requests.RequestException as e:
            logger.error(f"Error updating ticket {ticket_key}: {e}")
            return {"error": str(e)}
    
    def add_comment(self, ticket_key: str, comment: str) -> Dict:
        """
        Add a comment to a Jira ticket
        
        Args:
            ticket_key: The key of the Jira ticket
            comment: The comment text to add
            
        Returns:
            Dictionary containing the result
        """
        try:
            url = f"{self.base_url}/rest/api/3/issue/{ticket_key}/comment"
            
            data = {
                "body": {
                    "content": [
                        {
                            "content": [
                                {
                                    "text": comment,
                                    "type": "text"
                                }
                            ],
                            "type": "paragraph"
                        }
                    ],
                    "type": "doc",
                    "version": 1
                }
            }
            
            response = requests.post(url, auth=self.auth, headers=self.headers, json=data)
            response.raise_for_status()
            
            logger.info(f"Added comment to ticket {ticket_key}")
            return {"success": True}
            
        except requests.RequestException as e:
            logger.error(f"Error adding comment to ticket {ticket_key}: {e}")
            return {"error": str(e)}

class PRProcessingAgent(LlmAgent):
    """Agent responsible for processing PR approvals and merging"""
    
    def __init__(self, gitlab_toolkit: GitLabToolkit, jira_toolkit: JiraToolkit):
        super().__init__(
            name="pr_processing_agent",
            description="Processes GitLab PR approvals and handles merging",
            model="gemini-2.0-flash",  # Updated to use Gemini 2.0 Flash
            instruction="""
            You are a GitLab PR and Jira integration agent. Your responsibilities are:
            
            1. Monitor and process GitLab merge requests
            2. Check approval status before merging
            3. Merge approved PRs automatically
            4. Update corresponding Jira tickets
            5. Extract Jira ticket IDs from branch names
            6. Add comments and update ticket status after successful merges
            
            Always ensure PRs are properly approved before merging and handle errors gracefully.
            """,
            tools=[gitlab_toolkit, jira_toolkit]
        )
        
        # Store toolkits as instance variables after initialization
        object.__setattr__(self, 'gitlab_toolkit', gitlab_toolkit)
        object.__setattr__(self, 'jira_toolkit', jira_toolkit)
    
    def process_merge_requests(self) -> List[Dict]:
        """
        Main method to process all open merge requests
        
        Returns:
            List of processing results
        """
        results = []
        
        # Fetch open merge requests
        mrs = self.gitlab_toolkit.fetch_merge_requests("opened")
        
        for mr in mrs:
            try:
                result = self._process_single_mr(mr)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing MR {mr['iid']}: {e}")
                results.append({
                    "mr_id": mr['iid'],
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    def _process_single_mr(self, mr: Dict) -> Dict:
        """Process a single merge request"""
        mr_iid = mr['iid']
        
        # Get approval status
        approvals = self.gitlab_toolkit.get_merge_request_approvals(mr_iid)
        
        if not approvals:
            return {
                "mr_id": mr_iid,
                "status": "no_approvals_data",
                "message": "Could not fetch approval data"
            }
        
        # Check if MR is approved
        is_approved = (
            approvals.get('approved', False) or 
            len(approvals.get('approved_by', [])) >= approvals.get('approvals_required', 1)
        )
        
        if not is_approved:
            return {
                "mr_id": mr_iid,
                "status": "not_approved",
                "message": "MR is not yet approved"
            }
        
        # Check if MR can be merged
        if mr.get('merge_status') != 'can_be_merged':
            return {
                "mr_id": mr_iid,
                "status": "cannot_merge",
                "message": f"MR cannot be merged. Status: {mr.get('merge_status')}"
            }
        
        # Extract Jira ticket from branch name
        jira_ticket = self.gitlab_toolkit.extract_jira_ticket_from_branch(
            mr['source_branch']
        )
        
        # Merge the MR
        merge_result = self.gitlab_toolkit.merge_request(
            mr_iid,
            f"Merge {mr['title']} (closes #{mr_iid})"
        )
        
        if "error" in merge_result:
            return {
                "mr_id": mr_iid,
                "status": "merge_failed",
                "error": merge_result["error"]
            }
        
        result = {
            "mr_id": mr_iid,
            "status": "merged",
            "message": "MR successfully merged",
            "jira_ticket": jira_ticket
        }
        
        # Update Jira ticket if found
        if jira_ticket:
            jira_result = self._update_jira_ticket(jira_ticket, mr)
            result["jira_update"] = jira_result
        
        return result
    
    def _update_jira_ticket(self, ticket_key: str, mr: Dict) -> Dict:
        """Update Jira ticket after successful merge"""
        try:
            # Add comment about the merge
            comment = (
                f"🎉 Merge Request Successfully Merged!\n\n"
                f"**Title:** {mr['title']}\n"
                f"**Branch:** {mr['source_branch']} → {mr['target_branch']}\n"
                f"**URL:** {mr['web_url']}\n"
                f"**Merged at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"The code changes have been successfully integrated into the main branch."
            )
            
            comment_result = self.jira_toolkit.add_comment(ticket_key, comment)
            
            if comment_result.get("success"):
                # Try to move ticket to "Done" or "Resolved" status
                status_result = self.jira_toolkit.update_ticket_status(
                    ticket_key, 
                    "Done",
                    "Automatically updated after successful merge"
                )
                
                return {
                    "ticket_key": ticket_key,
                    "comment_added": True,
                    "status_updated": status_result.get("success", False),
                    "new_status": status_result.get("status", "unchanged")
                }
            else:
                return {
                    "ticket_key": ticket_key,
                    "comment_added": False,
                    "error": comment_result.get("error", "Unknown error")
                }
                
        except Exception as e:
            logger.error(f"Error updating Jira ticket {ticket_key}: {e}")
            return {
                "ticket_key": ticket_key,
                "error": str(e)
            }

class GitLabJiraAgent:
    """Main orchestration agent for GitLab-Jira integration"""
    
    def __init__(self):
        # Initialize toolkits with hardcoded values
        gitlab_toolkit = GitLabToolkit(
            base_url=GITLAB_BASE_URL,
            token=GITLAB_TOKEN,
            project_id=GITLAB_PROJECT_ID
        )
        
        jira_toolkit = JiraToolkit(
            base_url=JIRA_BASE_URL,
            username=JIRA_EMAIL,
            api_token=JIRA_API_TOKEN
        )
        
        # Initialize processing agent
        self.pr_agent = PRProcessingAgent(gitlab_toolkit, jira_toolkit)
    
    def run_processing_cycle(self) -> Dict:
        """Run a complete processing cycle"""
        logger.info("Starting GitLab-Jira processing cycle")
        
        try:
            results = self.pr_agent.process_merge_requests()
            
            # Summarize results
            total_processed = len(results)
            merged_count = len([r for r in results if r.get('status') == 'merged'])
            error_count = len([r for r in results if r.get('status') == 'error'])
            
            summary = {
                "timestamp": datetime.now().isoformat(),
                "total_processed": total_processed,
                "merged_count": merged_count,
                "error_count": error_count,
                "results": results
            }
            
            logger.info(f"Processing cycle completed: {merged_count}/{total_processed} merged")
            return summary
            
        except Exception as e:
            logger.error(f"Error in processing cycle: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

def main():
    """Main function to run the agent"""
    
    # Set environment variables for Google AI Studio (not Vertex AI)
    os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'FALSE'
    
    # Initialize and run agent
    agent = GitLabJiraAgent()
    
    # Run processing cycle
    result = agent.run_processing_cycle()
    
    # Print results
    print("\n" + "="*50)
    print("GITLAB-JIRA AGENT RESULTS")
    print("="*50)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()