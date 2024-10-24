"""
This file contains the regulatory document parsing function:

df = extract_subsections_from_pdf(path, preprosess = True)

The function takes in a pdf path to a regulatory document and outputs a dataframe of the subsection breakdowns of the document. 
This function makes a call to an LLM, chatgpt 3.5 turbo, to receive an example for a new subsection line. 
We then use this output to create a regular expression that can indicate when a new line is from a subsection split and when it is not. 
Using this information we parse the document into believed subsections and respond with the corresponding dataframe. 

This function depends on the TIKA parser tool and OPENAI API. 
If TIKA is unable to adequately parse the document into lines or if OPENAI is unable to notice new subsection lines, then this function will fail categorically. 
"""

import os
import re
import openai
import pandas as pd
import numpy as np
from tika import parser
from tqdm import tqdm
import random
import pytesseract
from pdf2image import convert_from_path

def query_LLM(system_role, usr_input):
    """
    system_role : str, the role of the system in the LLM query. 
    usr_input : str, the user input provided for the LLM query.
    output : str, the message content of the response from the gpt-3.5-turbo ChatGPT machine.
    
    An example could be: 
    Input:
    system_role : "You will be provided with two numbers separated by a comma. Your job is to add them together and present the result."
    usr_input : "3, 7"
    Output : "10"
    """
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_role},
        {"role": "user", "content": usr_input}]
    )
    return chat_completion.choices[0].message.content

def get_section_split_prefix_LLM(lines):
    # system_role = 'You will be provided with a list of lines from a parsed regulatory or municipal document. The lines will either indicate natural line breaks or new subsection breaks. You must provide an example of a line that begins a new subsection. You answer must be in the form: New Subsection: "Response". An example could be: Input: ["Chapter 1", "Description", "Section 1.01.010 Zoning", "How does this work?", "We do not know.", "Section 1.01.020 This is how it works."]. Output: New Subsection: "Section 1.01.020 This is". Another example could include: Input: ["§ 1.01.010. Adoption.", "Code adoption is the process of ", "§ 1.01.020. Title—Citation—Reference.", "Citation of Title References is"]. Output: New Subsection: "§ 1.01.010. Adoption."'
    
    system_role = """
    You will be provided with a list of lines from a parsed regulatory or municipal document. The lines will either indicate natural line breaks or new subsection breaks. You must provide an example of a line that begins a new subsection. \n
    
    You answer must be in the form: New Subsection: "Response". 
    \n
    Example_1 Input: 
    ["Chapter 1", "Description", "Section 1.01.010 Zoning", "How does this work?", "We do not know.", "Section 1.01.020 This is how it works."]. 
    \n
    Example_1 Output: 
    New Subsection: "Section 1.01.020 This is". 
    \n
    Example_2 Input: 
    ["§ 1.01.010. Adoption.", "Code adoption is the process of ", "§ 1.01.020. Title—Citation—Reference.", "Citation of Title References is"]. 
    \n
    Example_2 Output: 
    New Subsection: "§ 1.01.010. Adoption."
    """
    usr_input = ', '.join(lines)
    return query_LLM(system_role, usr_input)

def prepare_prefix_expression(input_line, strong_format = False):
    """
    input_line: str, this is a provided string that we are trying to make into a regular expresion. 

    output: str, the believed regular expression encoding of all subsection splits.
    """
    i = 0
    # Initialize the regular expression by stipulating that the pattern must be at the beginning of the line. That is what '^' means. 
    regex = "^"
    
    # Some of the subsections are noted by the primary symbol "§". 
    # So we will first check if the first character of the line is "§".
    if input_line[0] == "§":
        regex += "§\s*"
        i += 1
        while input_line[i] == ' ':
            i += 1

    # This checks if the first three characters are alphabetical in nature. Then adds an alphabetical pattern to the regular expression.
    elif re.search("^[sS][eE][cC][a-zA-Z]*", input_line):
        
        if re.search("^[sS][eE][cC][a-zA-Z]+", input_line): 
            regex += "[sS][eE][cC][a-zA-Z]+(\s*\.*)*" # If 'Section. ' is in the line.
        else: 
            regex += "[sS][eE][cC](\s*\.*)*" # If 'Sec. ' is in the line.
        
        while ((i < len(input_line)) and (input_line[i].isalpha())):
            i += 1
        
        while ((i < len(input_line)) and (input_line[i] in ' .')):
            i += 1

        # In the rare case, when we only see that the entire line is just one alphabetical string, we will get an error. 
        # It is likely that this will not be sufficient for our needs.
        if i == len(input_line):
            raise Exception(f"Error in handling line. Please check that line is valid.\n{input_line}")
    
    # regex_dash = "([0-9]+[A-Z]*-)+([0-9]*[A-Z]*\.*)+"
    # regex_no_dash = "([0-9]+[A-Z]*\.*)+(\(([0-9]*\.*)*\))*"

    while ((input_line[i].isalpha()) or (input_line[i].isnumeric()) or (input_line[i] in '.-:()')):
           i += 1
           
    regex += "(\(*[0-9]*[A-Z]*\.*\-*\:*\)*)*"
    
    return regex

def has_number(inputString):
    return bool(re.search(r'\d', inputString))

def get_subsection_line(lines):
    # Ask LLM to provide an example of a subsection split from the middle of the document. 
    num_chunks = 6
    subsection_line = ''
    num_iterations = 0

    def has_numbers(inputString):
        return any(char.isdigit() for char in inputString)
    
    while ((num_iterations < 5) and (len(subsection_line) == 0)):
        random_num = random.randint(1, num_chunks-1)
        lines_per_chunk = len(lines)//num_chunks
        threshold = min(100, lines_per_chunk)
        index_chunk_start = lines_per_chunk*random_num

        possible_subsection_line = get_section_split_prefix_LLM([x[:30] for x in lines[index_chunk_start:index_chunk_start+threshold]]).split('New Subsection: ')[1][1:-1]
        
        if has_numbers(possible_subsection_line[:20]) == True:
            subsection_line = possible_subsection_line
        else:
            num_iterations += 1
            num_chunks += 1
    
    if len(subsection_line) > 0:
        return subsection_line
    else: 
        return None

def split_regulatory_document_into_subsections(lines, template_subsection_line = None, strong_format = False):
    """
    This is a simple function which can parse many different regulatory documents. 
    It is highly dependent on the parser's performance. 
    If the parser does not provide an accurate line by line breakdown of the text it will not perform well. 
    It is assumed that the pdf contains only subsections from ONE source, in that the prefix for subsection splits will be consistent across the document. 
    If there are errors in formatting or incorrect subsection labels, this may impact the performance. 

    lines: array, the lines of a document after being parsed by something like the TIKA parser. 
    
    division: int, a divisor on the number of lines in the document. 
        If division == 2, we are looking half way into the document. 
        If division == 3, we are looking a third of the way into the document. 
        Each document may have an area where there is strange formatting that can cause invalid results. 
        This optional input allows you to change where you sample the document for example subsection splits. 
    
    output: pandas dataframe, this dataframe has columns = ['Subsection Number', 'Subsection_Name', 'Content']
    
    If this parser was unable to recover the accurate subsection splits first check the source document to see if a typo exists at that location. 
    Second, vary the division parameter as the parser may be unable to get an accurate subsection split candidate.
    The goal of this parser is to extract all of the subsection splits(and possibly more) that are in the document. 
    We can then filter the results and fix the formatting if there are errors. 
    If you wish to further provide breakdowns and add chapter/title information, this must be done separately. 
    """

    # The lines should be of the form...
    # file = parser.from_file(path_to_regulatory_pdf)
    # lines = [x for x in file['content'].split('\n') if len(x) > 0]   
    

    if template_subsection_line != None:
        subsection_line = template_subsection_line
    else:
        subsection_line = get_subsection_line(lines)

        if subsection_line is not None:
            pass
        else:
            raise Exception("Unable to get subsection split correctly please provide template subsection split manually.")
    
    # Create a regular expression of the format of this split to find possible subsection splits. 
    prepared_regex = prepare_prefix_expression(subsection_line, strong_format)
    # print(prepared_regex)
    
    # Get indices and content for each split line.
    subsection_splits = []
    for i, x in enumerate(lines):
        if re.search(prepared_regex, x):
            current_line = ' '.join([y for y in x.split(' ') if len(y) > 0]) # Clean the line from unwanted spaces before the prefix.
            # print(current_line)
            # subsection_number = re.search("[0-9]+\.([0-9]+\.)*[0-9]*", current_line).group()
            subsection_number = ''.join([y for y in re.search(prepared_regex, current_line).group() if y != '§']).lstrip().rstrip("-. ")
            # print(subsection_number)

            if has_number(subsection_number):
                subsection_splits.append([i,subsection_number])

    last_index = subsection_splits[-1][0]
    
    # Get content and add to dataframe. 
    Subsections = []
    for i, (subsection_split_index, subsection_number) in enumerate(subsection_splits):
        if subsection_split_index < last_index:
            subsection = lines[subsection_split_index: subsection_splits[i+1][0]]
        else:
            subsection = lines[subsection_split_index:]
        
        # print(subsection)

        subsection_name = subsection[0].split(subsection_number)[1]

        # We begin the subsection_name with the first alphabetic character if it is not None type.
        if re.search(r'[a-zA-Z]+', subsection_name) is not None:
            subsection_name = ' '.join([x for x in re.search(r'([a-zA-Z]+.*\s*)+\.*', subsection_name).group().split(" ") if len(x) > 0])
        else:
            subsection_name = ''
        
        content = ' '.join(subsection)
        Subsections.append([subsection_number, subsection_name, content])
        
    df = pd.DataFrame(Subsections, columns = ['subsection_number', 'subsection_name', 'content'])
    return df

def clean_subsection_dataframe(df, min_length = None):
    """
    df: pandas dataframe, the ouput of split_regulatory_document_into_subsections() function. 
        Has columns - ["subsection_number", "subsection_name", "content"]	
    min_length: number, every subsection should have at least this length when extracted. Something like 1.10.020 has length 8. If all subsections have at least this length, then this is our cutoff.	
    output: pandas dataframe, the "cleaned" dataframe. 

    This function is meant to go through the dataframe and fix mistakes in the split_regulatory_document_into_subsections() function. 
    It must be ambiguous to the formatting of the processed doc and should only make a change if there are "truths" we are using to fix them. 
    To that end, this function does the following:

    1. If the "subsection_name" is blank then we know there was no content other than the number provided in the subsection.
        This is the case only when the content of the believed subsection was only the subsection number itself. 
        Hence, there is essentially no content. In this case we delete the row.
    2. If the "subsection_name" begins with a lowercase letter, then the split was a mistake. 
        Subsection titles always begin with upper case letters. So a lower case letter indicates a mistaken split.
        In such a case we add this incorrect split to the previous line iteratively to go back to the last line that was correct. 
        The belief is that this previous line would be the correct subsection that had referenced the subsection from the incorrect split.
    """

    # This uses a mask to find all rows with a non-empty string as the "Subsection_Name".

    if len(df) == 1:
        # If there is only one row then even if it is incorrect, there is nothing we can do. 
        # In this case there are subsections that have no name but have a division/chapter name, but we miss this as we cannot parse for it. 
        pass 
    else:
        df = df[~(df["subsection_name"] == '')]
        
    df = df.reset_index()[["subsection_number", "subsection_name", "content"]]
    
    # Find splits that are indicated by a lower case letter leading the Subsection Name and possibly those whose subsection numbbers are not the correct length. 
    # In such a case we add these lines to the preceeding line that contained a correct Subsection Name.
    if min_length:
        line_error_mask = [True if ((x == '') or ((x[0].upper() == x[0]) and (len(y) >= min_length))) else False for x, y in df[["subsection_name", "subsection_number"]].values]
    else:
        line_error_mask = [True if ((x == '') or (x[0].upper() == x[0])) else False for x in df["subsection_name"]]

    # Note we iterate over lines.
    # If we have several lines in a row which are not valid splits, we add all of them to the preceeding correct line.  
    i = 1
    while i < len(line_error_mask):
        if line_error_mask[i] == True:
            i += 1
        else:
            j = 0
            while ((i + j < len(line_error_mask)) and (line_error_mask[i + j] == False)):
                df.loc[i-1, "content"] += df.loc[i + j, "content"]
                j += 1
            i += j
    
    # Then we use the mask to select only the valid lines. 
    df = df[line_error_mask]
    df = df.reset_index()[["subsection_number", "subsection_name", "content"]]
    
    # The next pass is not complete. 
    # It will check that the subsection numbers are in a strictly ascending order. 
    # This is not obvious as subsections can have letters, numbers, parentheses, and all manner of symbols.  
    # Any deviation from the desired ascending order is indicative of a mistake.
    # In such a case we will want to merge lines like above. 
    
    return df

def extract_subsections_from_pdf(path, template_subsection_line = None, strong_format = False, min_length = None, preprosess = False):
    """
    Wrapped function to take in a path and output the desired dataframe. 
    path: string, the path to the regulatory pdf file you wish to parse into subsections.
    
    min_length: number, every subsection should have at least this length when extracted. Something like 1.10.020 has length 8. If all subsections have at least this length, then this is our cutoff.	
    
    preprosess: boolean, if False we pass the content directly to the reference extraction without touching it past our initial split. 
        If True then we split the 'Content' at '\t' and take first part. Similarly, we split at '(Ord.' and take the first part if we do not break the sentences.
    
    output: pandas dataframe, the dataframe has columns
        ['Subsection_Number', 'Subsection_Name', 'Content', 'References', 'Referenced_By'] 
        or
        ['Subsection_Number', 'Subsection_Name', 'Content', 'Preprocessed_Content', 'References', 'Referenced_By']
        depending on if you specify to preprocess beforehand. 
    """

    # Use parser to parse the file then split it into lines.
    file = parser.from_file(path)

    lines = [x for x in file['content'].split('\n') if len(x) > 0] 

    df = split_regulatory_document_into_subsections(lines, template_subsection_line, strong_format)
    df = clean_subsection_dataframe(df, min_length)
    
    return df

def extract_subsections_from_pdf_using_OCR(path, template_subsection_line = None, strong_format = False, min_length = None, preprosess = False):
    """
    Wrapped function to take in a path and output the desired dataframe. 
    path: string, the path to the regulatory pdf file you wish to parse into subsections.
    
    min_length: number, every subsection should have at least this length when extracted. Something like 1.10.020 has length 8. If all subsections have at least this length, then this is our cutoff.	
    
    preprosess: boolean, if False we pass the content directly to the reference extraction without touching it past our initial split. 
        If True then we split the 'Content' at '\t' and take first part. Similarly, we split at '(Ord.' and take the first part if we do not break the sentences.
    
    output: pandas dataframe, the dataframe has columns
        ['Subsection_Number', 'Subsection_Name', 'Content', 'References', 'Referenced_By'] 
        or
        ['Subsection_Number', 'Subsection_Name', 'Content', 'Preprocessed_Content', 'References', 'Referenced_By']
        depending on if you specify to preprocess beforehand. 
    """

    # Use parser to parse the file then split it into lines.
    # This requires you to specify where the tessereact executable is. On my machine it is below. 
    # See this for details installing and calling tesseract: https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i
    # Download tesseract for windows here: https://github.com/UB-Mannheim/tesseract/wiki

    pytesseract.pytesseract.tesseract_cmd = f'C:\\Users\\jcste\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'

    pages = convert_from_path(path, 600)
    
    all_text = ''

    for page in pages:
        text = pytesseract.image_to_string(page)
        all_text += text + '\n'
        
    lines = [x for x in all_text.split('\n') if len(x) > 0] 

    df = split_regulatory_document_into_subsections(lines, template_subsection_line, strong_format)
    df = clean_subsection_dataframe(df, min_length)
    
    return df

def extract_subsections_from_lines(lines, template_subsection_line = None, strong_format = False, min_length = None, preprosess = False):
    """
    Wrapped function to take in a path and output the desired dataframe. 
    path: string, the path to the regulatory pdf file you wish to parse into subsections.
    preprosess: boolean, if False we pass the content directly to the reference extraction without touching it past our initial split. 
        If True then we split the 'Content' at '\t' and take first part. Similarly, we split at '(Ord.' and take the first part if we do not break the sentences.
    
    output: pandas dataframe, the dataframe has columns
        ['Subsection_Number', 'Subsection_Name', 'Content', 'References', 'Referenced_By'] 
        or
        ['Subsection_Number', 'Subsection_Name', 'Content', 'Preprocessed_Content', 'References', 'Referenced_By']
        depending on if you specify to preprocess beforehand. 
    """

    df = split_regulatory_document_into_subsections(lines, template_subsection_line, strong_format)
    df = clean_subsection_dataframe(df, min_length)
    
    return df






    