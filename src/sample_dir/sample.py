class Database:
    def __init__(self, connection_string):
        self.conn = connection_string

    def connect(self):
        print("Connecting...")

        def retry():
            print("Retrying connection...")
            return True

        return retry()

def standalone_function():
    return "I have no class"