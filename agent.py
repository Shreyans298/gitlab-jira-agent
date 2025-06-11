"""
GitLab PR & Jira Integration Agent using Google ADK

This agent:
1. Fetches PR requests from GitLab
2. Shows merge diffs in terminal
3. Asks for manual approval (Yes/No)
4. Merges approved PRs
5. Updates corresponding Jira tickets

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
            FunctionTool(self.get_merge_request_diff),
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
    
    def get_merge_request_diff(self, mr_iid: int) -> str:
        """
        Get the diff/changes for a specific merge request
        
        Args:
            mr_iid: The IID of the merge request
            
        Returns:
            String containing the diff content
        """
        try:
            url = f"{self.base_url}/api/v4/projects/{self.project_id}/merge_requests/{mr_iid}/changes"
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            changes = response.json()
            
            # Format the diff for terminal display
            diff_output = []
            diff_output.append(f"\n{'='*80}")
            diff_output.append(f"MERGE REQUEST DIFF - MR #{mr_iid}")
            diff_output.append(f"{'='*80}")
            
            if 'changes' in changes:
                for change in changes['changes']:
                    diff_output.append(f"\nFile: {change.get('new_path', change.get('old_path', 'Unknown'))}")
                    diff_output.append("-" * 60)
                    
                    if change.get('diff'):
                        diff_output.append(change['diff'])
                    else:
                        diff_output.append("Binary file or no changes to display")
                    
                    diff_output.append("")
            else:
                diff_output.append("No changes found in this merge request")
            
            diff_output.append(f"{'='*80}\n")
            
            diff_text = "\n".join(diff_output)
            logger.info(f"Fetched diff for MR {mr_iid}")
            return diff_text
            
        except requests.RequestException as e:
            logger.error(f"Error fetching diff for MR {mr_iid}: {e}")
            return f"Error fetching diff: {e}"
    
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
        Extract Jira ticket key from branch name for SUP project
        
        Args:
            branch_name: The name of the branch
            
        Returns:
            Jira ticket key if found, None otherwise
        """
        import re
        
        # Patterns specifically for SUP project
        patterns = [
            r'(SUP-\d+)',  # Standard format: SUP-123
            r'(SUP_\d+)',  # Underscore format: SUP_123
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
            FunctionTool(self.get_available_transitions),
            FunctionTool(self.update_ticket_status_by_transition_id),
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
            
            # Execute transition with proper format
            transition_data = {
                "transition": {
                    "id": target_transition['id']
                }
            }
            
            # Add comment in the correct format for API v3
            if comment:
                transition_data["update"] = {
                    "comment": [{
                        "add": {
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
                    }]
                }
            
            logger.info(f"Executing transition {target_transition['id']} ({target_transition['name']}) for ticket {ticket_key}")
            response = requests.post(transitions_url, auth=self.auth, 
                                   headers=self.headers, json=transition_data)
            response.raise_for_status()
            
            logger.info(f"Updated ticket {ticket_key} to status '{status}'")
            return {"success": True, "status": status}
            
        except requests.RequestException as e:
            logger.error(f"Error updating ticket {ticket_key}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    logger.error(f"JIRA API Error Details: {error_details}")
                    return {"error": f"{str(e)} - Details: {error_details}"}
                except:
                    return {"error": str(e)}
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
    
    def get_available_transitions(self, ticket_key: str) -> Dict:
        """
        Get available transitions for a Jira ticket (for debugging)
        
        Args:
            ticket_key: The key of the Jira ticket
            
        Returns:
            Dictionary containing available transitions
        """
        try:
            url = f"{self.base_url}/rest/api/3/issue/{ticket_key}/transitions"
            
            response = requests.get(url, auth=self.auth, headers=self.headers)
            response.raise_for_status()
            
            transitions_data = response.json()
            transitions = []
            
            for transition in transitions_data.get('transitions', []):
                transitions.append({
                    'id': transition['id'],
                    'name': transition['name'],
                    'to_status': transition['to']['name']
                })
            
            logger.info(f"Available transitions for {ticket_key}: {transitions}")
            return {
                "ticket_key": ticket_key,
                "transitions": transitions
            }
            
        except requests.RequestException as e:
            logger.error(f"Error fetching transitions for {ticket_key}: {e}")
            return {"error": str(e)}
    
    def update_ticket_status_by_transition_id(self, ticket_key: str, transition_id: str, comment: str = None) -> Dict:
        """
        Update ticket status using transition ID directly
        
        Args:
            ticket_key: The key of the Jira ticket
            transition_id: The ID of the transition to execute
            comment: Optional comment to add
            
        Returns:
            Dictionary containing update result
        """
        try:
            transitions_url = f"{self.base_url}/rest/api/3/issue/{ticket_key}/transitions"
            
            # Execute transition with minimal data
            transition_data = {
                "transition": {
                    "id": transition_id
                }
            }
            
            logger.info(f"Executing transition ID {transition_id} for ticket {ticket_key}")
            response = requests.post(transitions_url, auth=self.auth, 
                                   headers=self.headers, json=transition_data)
            response.raise_for_status()
            
            logger.info(f"Successfully executed transition {transition_id} for ticket {ticket_key}")
            return {"success": True, "transition_id": transition_id}
            
        except requests.RequestException as e:
            logger.error(f"Error executing transition {transition_id} for ticket {ticket_key}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    logger.error(f"JIRA API Error Details: {error_details}")
                    return {"error": f"{str(e)} - Details: {error_details}"}
                except:
                    return {"error": str(e)}
            return {"error": str(e)}

class PRProcessingAgent(LlmAgent):
    """Agent responsible for processing PR approvals and merging"""
    
    def __init__(self, gitlab_toolkit: GitLabToolkit, jira_toolkit: JiraToolkit):
        super().__init__(
            name="pr_processing_agent",
            description="Processes GitLab PR approvals and handles merging with manual approval",
            model="gemini-2.0-flash",  # Updated to use Gemini 2.0 Flash
            instruction="""
            You are a GitLab PR and Jira integration agent. Your responsibilities are:
            
            1. Monitor and process GitLab merge requests
            2. Display merge request diffs in terminal
            3. Ask for manual approval before merging
            4. Merge approved PRs automatically
            5. Update corresponding Jira tickets (SUP project)
            6. Extract Jira ticket IDs from branch names
            7. Add comments and update ticket status after successful merges
            
            Always show diffs and ask for manual approval before merging.
            """,
            tools=[gitlab_toolkit, jira_toolkit]
        )
        
        # Store toolkits as instance variables after initialization
        object.__setattr__(self, 'gitlab_toolkit', gitlab_toolkit)
        object.__setattr__(self, 'jira_toolkit', jira_toolkit)
    
    def process_merge_requests(self) -> List[Dict]:
        """
        Main method to process all open merge requests with manual approval
        
        Returns:
            List of processing results
        """
        results = []
        
        # Fetch open merge requests
        mrs = self.gitlab_toolkit.fetch_merge_requests("opened")
        
        if not mrs:
            print("\n🔍 No open merge requests found.")
            return []
        
        print(f"\n📋 Found {len(mrs)} open merge request(s)")
        
        for i, mr in enumerate(mrs, 1):
            try:
                print(f"\n{'='*80}")
                print(f"PROCESSING MERGE REQUEST {i}/{len(mrs)}")
                print(f"{'='*80}")
                
                result = self._process_single_mr(mr)
                results.append(result)
                
                # If user rejected, stop processing remaining MRs
                if result.get('status') == 'user_rejected':
                    print(f"\n⏹️  Stopping processing of remaining merge requests.")
                    break
                    
            except Exception as e:
                logger.error(f"Error processing MR {mr['iid']}: {e}")
                results.append({
                    "mr_id": mr['iid'],
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    def _process_single_mr(self, mr: Dict) -> Dict:
        """Process a single merge request with manual approval"""
        mr_iid = mr['iid']
        
        print(f"\n📄 MR #{mr_iid}: {mr['title']}")
        print(f"👤 Author: {mr['author']['name']}")
        print(f"🌿 Source: {mr['source_branch']} → {mr['target_branch']}")
        print(f"🔗 URL: {mr['web_url']}")
        
        # Extract Jira ticket from branch name
        jira_ticket = self.gitlab_toolkit.extract_jira_ticket_from_branch(
            mr['source_branch']
        )
        
        if jira_ticket:
            print(f"🎫 Jira Ticket: {jira_ticket}")
        else:
            print("⚠️  No Jira ticket found in branch name")
        
        # Check if MR can be merged
        if mr.get('merge_status') != 'can_be_merged':
            print(f"❌ Cannot merge: {mr.get('merge_status')}")
            return {
                "mr_id": mr_iid,
                "status": "cannot_merge",
                "message": f"MR cannot be merged. Status: {mr.get('merge_status')}"
            }
        
        # Get and display the diff
        print(f"\n🔍 Fetching merge request diff...")
        diff = self.gitlab_toolkit.get_merge_request_diff(mr_iid)
        print(diff)
        
        # Ask for manual approval
        while True:
            try:
                approval = input("🤔 Do you want to merge this MR? (yes/y/no/n): ").strip().lower()
                if approval in ['yes', 'y']:
                    approved = True
                    break
                elif approval in ['no', 'n']:
                    approved = False
                    break
                else:
                    print("⚠️  Please enter 'yes', 'y', 'no', or 'n'")
            except KeyboardInterrupt:
                print(f"\n\n⏹️  Process interrupted by user")
                return {
                    "mr_id": mr_iid,
                    "status": "interrupted",
                    "message": "Process interrupted by user"
                }
        
        if not approved:
            print(f"❌ MR #{mr_iid} rejected by user")
            return {
                "mr_id": mr_iid,
                "status": "user_rejected",
                "message": "MR rejected by user"
            }
        
        # Merge the MR
        print(f"\n🔄 Merging MR #{mr_iid}...")
        merge_result = self.gitlab_toolkit.merge_request(
            mr_iid,
            f"Merge {mr['title']} (closes #{mr_iid})"
        )
        
        if "error" in merge_result:
            print(f"❌ Merge failed: {merge_result['error']}")
            return {
                "mr_id": mr_iid,
                "status": "merge_failed",
                "error": merge_result["error"]
            }
        
        print(f"✅ MR #{mr_iid} successfully merged!")
        
        result = {
            "mr_id": mr_iid,
            "status": "merged",
            "message": "MR successfully merged",
            "jira_ticket": "SUP-125"  # Hardcoded ticket
        }
        
        # Always update the hardcoded Jira ticket SUP-125
        print(f"🎫 Updating Jira ticket SUP-125...")
        jira_result = self._update_jira_ticket("SUP-125", mr)
        result["jira_update"] = jira_result
        
        if jira_result.get("comment_added") or jira_result.get("status_updated"):
            print(f"✅ Jira ticket SUP-125 updated successfully!")
        else:
            print(f"⚠️  Failed to update Jira ticket SUP-125: {jira_result.get('error', 'Unknown error')}")
        
        return result
    
    def _update_jira_ticket(self, ticket_key: str, mr: Dict) -> Dict:
        """Update Jira ticket after successful merge"""
        try:
            # First, check what transitions are available for debugging
            print(f"🔍 Checking available transitions for {ticket_key}...")
            transitions_info = self.jira_toolkit.get_available_transitions(ticket_key)
            
            if transitions_info.get("error"):
                print(f"❌ Error checking transitions: {transitions_info['error']}")
                return {
                    "ticket_key": ticket_key,
                    "error": f"Cannot access ticket transitions: {transitions_info['error']}"
                }
            
            available_transitions = transitions_info.get("transitions", [])
            print(f"📋 Available transitions for {ticket_key}:")
            for transition in available_transitions:
                print(f"   • {transition['name']} → {transition['to_status']}")
            
            # Add comment about the merge
            comment = (
                f"🎉 Merge Request Successfully Merged!\n\n"
                f"**Title:** {mr['title']}\n"
                f"**Branch:** {mr['source_branch']} → {mr['target_branch']}\n"
                f"**URL:** {mr['web_url']}\n"
                f"**Merged at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"The code changes have been successfully integrated into the main branch."
            )
            
            print(f"📝 Adding comment to {ticket_key}...")
            comment_result = self.jira_toolkit.add_comment(ticket_key, comment)
            
            # Try to find a suitable completion status from available transitions
            completion_statuses = ["Done", "Completed", "Closed", "Resolved", "Complete"]
            status_result = {"success": False}
            
            # First, try to find "Mark as done" transition specifically (ID 61 from the output)
            mark_as_done_transition = None
            for transition in available_transitions:
                if transition['name'].lower() == 'mark as done' or transition['to_status'].lower() == 'done':
                    mark_as_done_transition = transition
                    break
            
            if mark_as_done_transition:
                print(f"🔄 Using direct transition '{mark_as_done_transition['name']}' (ID: {mark_as_done_transition['id']})...")
                status_result = self.jira_toolkit.update_ticket_status_by_transition_id(
                    ticket_key, 
                    mark_as_done_transition['id']
                )
                if status_result.get("success"):
                    print(f"✅ Successfully executed transition '{mark_as_done_transition['name']}'")
                else:
                    print(f"⚠️  Failed to execute transition '{mark_as_done_transition['name']}': {status_result.get('error', 'Unknown error')}")
            
            # If direct transition failed, try the old method
            if not status_result.get("success"):
                for status in completion_statuses:
                    # Check if this status is available in transitions
                    matching_transition = None
                    for transition in available_transitions:
                        if transition['to_status'].lower() == status.lower():
                            matching_transition = transition
                            break
                    
                    if matching_transition:
                        print(f"🔄 Updating {ticket_key} status to '{matching_transition['to_status']}'...")
                        status_result = self.jira_toolkit.update_ticket_status(
                            ticket_key, 
                            matching_transition['to_status'],
                            "Automatically updated after successful merge"
                        )
                        if status_result.get("success"):
                            print(f"✅ Successfully updated status to '{matching_transition['to_status']}'")
                            break
                        else:
                            print(f"⚠️  Failed to update to '{matching_transition['to_status']}': {status_result.get('error', 'Unknown error')}")
            
            if not status_result.get("success"):
                print(f"⚠️  Could not find suitable completion status. Available transitions:")
                for transition in available_transitions:
                    print(f"     • {transition['name']} → {transition['to_status']}")
            
            return {
                "ticket_key": ticket_key,
                "comment_added": comment_result.get("success", False),
                "status_updated": status_result.get("success", False),
                "new_status": status_result.get("status", status_result.get("transition_id", "unchanged")),
                "available_transitions": [t['to_status'] for t in available_transitions],
                "comment_error": comment_result.get("error") if not comment_result.get("success") else None,
                "status_error": status_result.get("error") if not status_result.get("success") else None
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
    
    os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'FALSE'
    
    print("🚀 Starting GitLab-Jira Integration Agent")
    print("This agent will:")
    print("  1. Fetch open merge requests from GitLab")
    print("  2. Show merge diffs in terminal")
    print("  3. Ask for your approval before merging")
    print("  4. Update corresponding SUP Jira tickets")
    print(f"{'='*80}")
    
    # Initialize and run agent
    agent = GitLabJiraAgent()
    
    # Run processing cycle
    result = agent.run_processing_cycle()
    
    # Print final results
    print("\n" + "="*80)
    print("🏁 GITLAB-JIRA AGENT FINAL RESULTS")
    print("="*80)
    
    if result.get("error"):
        print(f"❌ Error occurred: {result['error']}")
    else:
        print(f"📊 Total processed: {result.get('total_processed', 0)}")
        print(f"✅ Successfully merged: {result.get('merged_count', 0)}")
        print(f"❌ Errors: {result.get('error_count', 0)}")
        
        # Show detailed results
        for mr_result in result.get('results', []):
            mr_id = mr_result.get('mr_id')
            status = mr_result.get('status')
            
            if status == 'merged':
                print(f"  ✅ MR #{mr_id}: Merged successfully")
                if mr_result.get('jira_ticket'):
                    jira_status = "✅" if mr_result.get('jira_update', {}).get('comment_added') else "⚠️"
                    print(f"    {jira_status} Jira: {mr_result['jira_ticket']}")
            elif status == 'user_rejected':
                print(f"  ❌ MR #{mr_id}: Rejected by user")
            elif status == 'cannot_merge':
                print(f"  ⚠️  MR #{mr_id}: Cannot be merged")
            elif status == 'error':
                print(f"  ❌ MR #{mr_id}: Error - {mr_result.get('error', 'Unknown error')}")
    
    print(f"{'='*80}")
    print("🎉 GitLab-Jira Agent completed!")

if __name__ == "__main__":
    main()