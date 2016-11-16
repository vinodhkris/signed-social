# signed-social
This is an implementation of the tagger and augmented dataset developed for 

'Krishnan, Vinodh, and Jacob Eisenstein. "You're Mr. Lebowski, I'm the Dude.": Inducing Address Term Formality in Signed Social Networks'
[http://www.aclweb.org/anthology/N/N15/N15-1185.pdf]

This project has 3 folders

1. Cornell-movie-dialogs-data : Which contains the dataset for this project. This is taken directly from the Cornell-movie-dialogs dataset from the website.

2. Dataset_with_character_full_names - This contains the modified file for the 'movie_characters_metadata.txt' which contains the full names for each of the characters. This is obtained by extracting from rottenTomatoes.com (code in util.py in the tagger folder)

3. Tagger - This is the tagger that is employed to obtain the final set of address terms, and determine which address term was used for the addressee from the movie dialog sentences. The tagger trained on the movie dialogs dataset instances is in 'trained_tagger.pickle'. 'bilouTagger.py' contains code on how to use it. 

Please contact me at krishnan.vinodh@gmail.com if there are any issues.

