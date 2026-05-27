from pathlib import Path

def main():

    # Firstly, lets import all video files from UCF101 and put them into a list
    path = Path('/Users/darcysprigg/Coding/Co-op summer 2026/UCF101')
    video_files = list(path.rglob('*.avi'))
    # print(video_files)

if __name__ == "__main__":
    main()