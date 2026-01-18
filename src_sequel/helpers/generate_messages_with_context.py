import json

def message_to_string_format(data, include_immediate_response=True):
    """
    Convert JSON conversations to "author: message" string format
    
    Args:
        json_filepath: Path to JSON file
        include_immediate_response: If True, includes the immediate response message
                                   (the next message from a different author)
    
    Returns:
        tuple: (formatted_strings, response_strings)
               response_strings will be empty list if include_immediate_response=False
    """
    
    # Extract conversations
    if 'conversations' in data:
        conversations = data['conversations']
    elif 'selected_conversations' in data:
        conversations = data['selected_conversations']
    else:
        conversations = data  # Assume it's a list
    
    formatted_strings = []
    response_strings = []
    
    for conv in conversations:
        lines = []
        
        # Add context messages
        for msg in conv.get('context_messages', []):
            author = msg.get('Author', 'Unknown')
            content = msg.get('Content', '')
            lines.append(f"{author}: {content}")
        
        # Add current message
        curr_msg = conv.get('current_message', {})
        author = curr_msg.get('Author', 'Unknown')
        content = curr_msg.get('Content', '')
        lines.append(f"{author}: {content}")
        
        # Store the formatted context + current
        context_current_str = "\n".join(lines)
        formatted_strings.append(context_current_str)
        
        # If we want the immediate response, add it
        if include_immediate_response:
            # Check if there's an immediate response in the JSON
            if 'immediate_response' in conv:
                resp_msg = conv['immediate_response']
                resp_author = resp_msg.get('Author', 'Unknown')
                resp_content = resp_msg.get('Content', '')
                response_strings.append(f"{resp_author}: {resp_content}")
            else:
                response_strings.append("")  # Empty string if no response
    
    return formatted_strings