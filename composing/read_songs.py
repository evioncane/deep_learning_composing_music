import glob, os

songs_path = '/home/evioncane/Git/deep_learning_composing_music/input_songs'

os.chdir(songs_path)


def read_songs():
     songs = []
     for file_name in glob.glob("**/*.abc", recursive=True):
          file = open(file_name, 'r')
          song = file.read()
          songs.append(song)
     return songs