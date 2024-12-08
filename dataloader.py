import os

class DataLoader:
    def __init__(self, file_paths, chunk_size=1000):
        """
        Initializes the DataLoader for multiple log files.

        :param file_paths: List of log file paths to process.
        :param chunk_size: The number of log lines to yield in each chunk.
        """
        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.files = [open(file_path, 'r') for file_path in file_paths]  # Open all files
        self.current_file_idx = 0
        self.current_file = self.files[self.current_file_idx]

    def __iter__(self):
        """
        Returns an iterator to process the log files in chunks.
        """
        return self

    def __next__(self):
        """
        Reads the next chunk of log lines from the current file.

        :return: A list of log lines as a chunk.
        """
        lines = []
        while len(lines) < self.chunk_size:
            line = self.current_file.readline()
            if not line:  # If end of file is reached, move to next file
                self.current_file_idx += 1
                if self.current_file_idx >= len(self.files):
                    # If we've processed all files, stop iteration
                    raise StopIteration
                self.current_file = self.files[self.current_file_idx]
                continue
            lines.append(line.strip())  # Strip newlines and whitespace from each line
        return self.current_file_idx, lines

    def close(self):
        """
        Close all open files when finished.
        """
        for file in self.files:
            file.close()
