import json
import traceback

# Remove global variables since they'll be managed in state
# question_index = 0
# current_node = None

# Load the JSON data
JSON_FILE = "data/conversation-tree-v2.json"
# Preload the JSON data at module level
try:
    with open(JSON_FILE, 'r', encoding='utf-8') as file:
        JSON_DATA = json.load(file)
except FileNotFoundError:
    print(f"❌ Error: JSON File '{JSON_FILE}' not found.")
    JSON_DATA = {}
except json.JSONDecodeError as e:
    print(f"❌ Error: Invalid JSON format - {e}")
    JSON_DATA = {}
except Exception as e:
    print(f"❌ Error: {e}")
    JSON_DATA = {}
    
def get_question(speaker_name, index=0):
    """
    Return message and audio for a speaker from the first level of questions.
    
    Args:
        speaker_name (str): Name of the speaker (e.g., 'student_1', 'teacher_1')
        index (int): Question index to use
    
    Returns:
        dict: Dictionary containing 'message' and 'audio' for the speaker, or None if no more questions
    """
    results = []
    
    if 'questions' in JSON_DATA:
        # Check if the index is valid
        if index >= len(JSON_DATA['questions']):
            return None  # No more questions available
            
        initial_question = JSON_DATA['questions'][index]
        for question in initial_question['children']:
            print(question.get('speaker'))
            if question.get('speaker') == speaker_name:
                result = {
                    'message': question.get('message', ''),
                    'question': initial_question.get('message', ''),
                    'criteria': initial_question.get('criteria', ''),
                    'audio': question.get('audio link', ''),
                    'responses': question.get('responses', [])
                }
                return result
    return {
        'message':'Speaker not found',
        'audio': ''
    }

def find_speaker_responses(data, speaker_name):
    """
    Find the first node that matches the speaker_name by traversing children recursively.
    
    Args:
        data: The JSON data structure (can be a dict or list)
        speaker_name: The speaker name to search for
        
    Returns:
        The first matching node or None if not found
    """
    if isinstance(data, dict):
        # get the responses of this node
        if 'responses' in data:
            data = data['responses']    
    if isinstance(data, list):        # Search through list items
        nodes = []
        for item in data:
            if 'responses' in item:
                for child in item['responses']:
                    if child.get('speaker') == speaker_name:
                        nodes.append(child)
        return nodes
    
    return None

def build_response_prompt(speaker_name, state):
    """
    Return prompt to choose the best response from the node.
    
    Args:
        speaker_name (str): Name of the speaker
        state (dict): Current conversation state containing question_index and current_node
    """
    try:
        # Use state instead of global variables
        question_index = state.get("question_index", 0)
        current_node = state.get("current_node")

        question_node = get_question(speaker_name, question_index)
        
        # Check if we've run out of questions
        if question_node is None:
            return None, None, None

        if current_node is None:
            current_node = get_question(speaker_name, question_index)
            state["current_node"] = current_node  # Update state

        if 'responses' not in current_node:
            # No more responses, end of conversation
            return None, None, None

        response_nodes = find_speaker_responses(current_node['responses'], speaker_name)
        response_messages = []
        tags = []
        for node in response_nodes:
            response_messages.append(node['message'])
            tags.append(node['tag'])

        # build the prompt
        question_or_message = current_node.get("question") or current_node.get("message")

        prompt = f""" You are a teacher doing an assessment of a student's response. Choose the best response from the list of responses based on the criteria. Check if response answers the question. When student do not know the answer, do not say correct. 
        Question: {question_or_message}
        Criteria: {question_node['criteria']}
        Responses: {response_messages}

        Respond verbatim from the response given above. Do not add any other text.
        """
        return (prompt, response_messages, tags)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Stack trace:")
        traceback.print_exc()
        return None, None, None
        

def find_node_by_tag(tag):
    """
    Find the node with the given tag from the JSON data.
    
    Args:
        tag (str): The tag to search for
        
    Returns:
        dict: The node with the matching tag, or None if not found
    """
    def search_recursive(data):
        """Recursively search through the data structure for the tag."""
        if isinstance(data, dict):
            if data.get('tag') == tag:
                return data
            # Search in responses if they exist
            if 'responses' in data:
                for response in data['responses']:
                    result = search_recursive(response)
                    if result:
                        return result
            # Search in children if they exist
            if 'children' in data:
                for child in data['children']:
                    result = search_recursive(child)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = search_recursive(item)
                if result:
                    return result
        return None
    
    # Start the search from the questions
    if 'questions' in JSON_DATA:
        for question in JSON_DATA['questions']:
            result = search_recursive(question)
            if result:
                return result
    
    return None