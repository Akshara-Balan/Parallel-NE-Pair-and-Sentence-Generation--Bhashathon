# src/ne_alignment.py
import spacy
import stanza
from awesome_align.run_align import align_text  # Import for word alignment

# Load English and target NER models
nlp_en = spacy.load("en_core_web_sm")
nlp_target = stanza.Pipeline(lang="ml", processors="tokenize,ner")  # Change lang as needed

def extract_named_entities(sentence, lang="en"):
    """Extract Named Entities from a sentence."""
    if lang == "en":
        doc = nlp_en(sentence)
        return [(ent.text, ent.label_) for ent in doc.ents]
    else:
        doc = nlp_target(sentence)
        return [(ent.text, ent.type) for ent in doc.ents]

def align_entities(en_sentence, target_sentence):
    """Align Named Entities using a neural aligner."""
    aligned_text = align_text([en_sentence], [target_sentence])
    return aligned_text

if __name__ == "__main__":
    en_sentence = "John Doe went to Paris last year."
    target_sentence = "ജോൺ ഡോ കഴിഞ്ഞ വർഷം പാരീസിലേക്ക് പോയി."

    en_entities = extract_named_entities(en_sentence, "en")
    target_entities = extract_named_entities(target_sentence, "ml")

    print(f"English NEs: {en_entities}")
    print(f"Target NEs: {target_entities}")
    print(f"Aligned NEs: {align_entities(en_sentence, target_sentence)}")
