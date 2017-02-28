## Synopsis

This project uses Sung Kim's Word-Rnn-Tensorflow (https://github.com/hunkim/word-rnn-tensorflow), which uses code from https://github.com/sherjilozair/char-rnn-tensorflow which was inspired from Andrej Karpathy's char-rnn. Our UI generates text from a set of pretrained inputs, and the user can switch between inputs as they add more text.

## Installation

This project requires installing Python and Django.

To use the UI, cd into the "ui" directory and run

```sh
python manage.py runserver
```

Then navigate to your local host on your browser and add /ui
(ex: http://localhost:8000/ui/)

## Instructions for UI

Select an input from the dropdown menu from which text will be generated. Choose the number of words to add.

Click "generate text".


Each time "generate text" is clicked, text will be added to what has already been displayed, and the already generated text will be used to prime the next generation. When no text has yet been generated, a random word will be chosen from the input to prime the generation.  Click "clear output" to start over.
