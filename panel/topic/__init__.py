"""A topic: its vocabulary, the framings that word it per language, and the response models built from both."""
from .vocabulary import Reference, Topic, load_topic, topic_names
from .framing import Framing, framing_names, load_framing
from .question import Question, load_question, question_names
from .response import make_response_models
