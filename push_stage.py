import os
import sys
from snowflake.snowpark import Session

def list_files_and_folders(path):
    result = []
    for root, dirs, files in os.walk(path):
        # Exclude directories starting with a dot or underscore
        dirs[:] = [d for d in dirs if not (d.startswith('.') or d.startswith('_'))]
        for file in files:
            result.append(os.path.join(root, file))
    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide the password as an argument.")
        sys.exit(1)

    password = sys.argv[1]

    # Usage example
    directory_path = os.getcwd()
    result = list_files_and_folders(directory_path)

    connection_params = {
        'account': 'hbb49926',
        'user': 'madkins',
        'password': password,
        'role': 'SERVICESNOW_USER_ROLE',
        'warehouse': 'MADKINS',
        'database': 'SPCS_DEMO',
        'schema': 'PROD'
    }

    session = Session.builder.configs(connection_params).create()

    query = '''
        create or replace stage snowpark_container_demo
            ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
    '''

    #session.sql(query).collect()

    for file in result:
        query = f'''
            put file://{file} @snowpark_container_demo{file[:file.rfind('/')].replace(os.getcwd(),'')}
            auto_compress=false overwrite=true;
        '''
        print(file)
        session.sql(query).collect()
