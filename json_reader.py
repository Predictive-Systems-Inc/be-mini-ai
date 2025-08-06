import json

# Preload the JSON data at module level
try:
    with open("conversation-tree-simplified-with-audio.json", 'r', encoding='utf-8') as file:
        JSON_DATA = json.load(file)
except FileNotFoundError:
    print("❌ Error: File 'conversation-tree-simplified-with-audio.json' not found.")
    JSON_DATA = {}
except json.JSONDecodeError as e:
    print(f"❌ Error: Invalid JSON format - {e}")
    JSON_DATA = {}
except Exception as e:
    print(f"❌ Error: {e}")
    JSON_DATA = {}
    
def get_first_question(speaker_name):
    """
    Return message and audio for a speaker from the first level of questions.
    
    Args:
        speaker_name (str): Name of the speaker (e.g., 'student_1', 'teacher_1')
    
    Returns:
        dict: Dictionary containing 'message' and 'audio' for the speaker
    """
    results = []
    
    if 'questions' in JSON_DATA:
        initial_question = JSON_DATA['questions'][0]
        for question in initial_question['children']:
            print(question.get('speaker'))
            if question.get('speaker') == speaker_name:
                result = {
                    'message': question.get('message', ''),
                    'question': initial_question.get('message', ''),
                    'criteria': initial_question.get('criteria', ''),
                    'audio': question.get('audio link', ''),
                }
                return result
    return {
        'message':'Speaker not found',
        'audio': ''
    }

def find_first_speaker_node(data, speaker_name):
    """
    Find the first node that matches the speaker_name by traversing children recursively.
    
    Args:
        data: The JSON data structure (can be a dict or list)
        speaker_name: The speaker name to search for
        
    Returns:
        The first matching node or None if not found
    """
    if isinstance(data, dict):
        return None
                    
    elif isinstance(data, list):        # Search through list items
        nodes = []
        for item in data:
            if 'children' in item:
                for child in item['children']:
                    if child.get('speaker') == speaker_name:
                        nodes.append(child)
        return nodes
    
    return None
    
def build_response_prompt(speaker_name, current_node):
    """
    Return prompt to choose the best response from the node.
    """
    print("starting build_response_prompt...")
    # find the question
    question_node = [question for question in JSON_DATA['questions'] if question['message'] == current_node['question']][0]

    # print("current node!!!!!", current_node)

    search_node = question_node
    # if 'children' in current_node and current_node['children']:
    #     print("current node has children, using for search")
    #     search_node = current_node
    # find the responses for the speaker
    responses = []
    for child in search_node['children']:
        if child['speaker'] == speaker_name:   
            response_nodes = find_first_speaker_node(child['children'], speaker_name)
            if response_nodes:  # Only add if we found valid responses
                # If response_nodes is a list, extend responses with it
                if isinstance(response_nodes, list):
                    responses.extend(response_nodes)
                else:
                    responses.append(response_nodes)

    response_messages = []
    for response in responses:
        response_messages.append(response['message'])

    # build the prompt

    prompt = f""" You are a tutor assessing a student's response. Choose the best response from the list of responses based on the criteria. Respond verbatim from the response.
    Question: {current_node['question']}
    Criteria: {current_node['criteria']}
    Responses: {response_messages}
    """

    print("prompt", prompt)
    return (prompt, responses)
