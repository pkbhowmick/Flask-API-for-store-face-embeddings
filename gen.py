import uuid

def getId():
    try:
        myfile = open("uuid.txt","w")
        id = uuid.uuid1
        myfile.write(str(id))

    except Exception as e:
        print(str(e))