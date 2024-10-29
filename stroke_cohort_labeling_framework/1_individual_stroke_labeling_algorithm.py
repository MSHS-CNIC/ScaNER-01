"""Step1: Main NLP algorithm to label records.
"""

import spacy
from spacy import displacy
from spacy.tokens import Span
from spacy.tokenizer import Tokenizer
from spacy.matcher import Matcher
from spacy.util import compile_prefix_regex, compile_suffix_regex

from collections import defaultdict

from negspacy.negation import Negex
from negspacy.termsets import termset

from wordsegment import load, segment

from enum import Enum

import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from util.data import load_accession_data


PATHOLOGIES = ["cva", "hematoma", "edema", "hemorrhage", "hemorrhagic", "stroke", "ischemic", "ischemia", "infarct", "infarction", "lacunar", "lacune"]
ACUTE_DESCS = ["acute", "recent", "new", "subacute", "early"]
NON_ACUTE_DESCS = ["old", "chronic", "made", "persistent", "marked"]
WORSE_DESCS = ["developed", "demarcation", "progression", "progressive", "expansion", "expanded", "worsening", "worsened", "enlarging", "enlarged", "increased"]
REGIONS = ["microvascular", "subarachnoid", "subdural", "parenchymal", "mca", "aca", "pca", "ica", "intracranial", "extracranial", "cervical", "lobar"]
CERTAINTIES = ["unlikely", "uncertain", "likely", "probable", "possible", "suspected", "suspicious", "definitive"]
DENSITIES = ["dense", "isodense", "hypodense", "hyperdense"]

# Step0: Correct typos/mashed words
load()

def custom_segment(text, keep_as_is_lists):
    # Flatten all lists into a single set for efficient lookup
    keep_as_is_set = {word for lst in keep_as_is_lists for word in lst}

    # Separate words from numbers without splitting numbers
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)

    # Split by spaces and punctuation
    words = re.findall(r'\b\w+\b', text)

    # Process segments with wordsegment
    corrected_words = []
    for word in words:
        if word.lower() in keep_as_is_set or not word.isalpha():
            corrected_words.append(word)
        else:
            corrected_words.append(' '.join(segment(word)))

    # Rejoin words with original spacing and punctuation
    corrected_text = text
    for word, corrected_word in zip(words, corrected_words):
        corrected_text = corrected_text.replace(word, corrected_word, 1)

    return corrected_text


# Step1: If there is a period between two strings with no space after the period, insert one.  
def insert_space_after_period_percent(text):
    text = re.sub(r'\.(?=[^\s])', '. ', text)
    return re.sub(r'%(?=[^\s%])', '% ', text)

# Step2: Split the report text into findings and impression
def extract_findings_impression(text):
    findings_text = None
    impression_text = None

    # Split on "FINDINGS:"
    findings_split = re.split(r'FINDINGS\s*[:]\s*', text, maxsplit=1, flags=re.IGNORECASE)
    
    if len(findings_split) > 1:
        # Split the remaining content on "IMPRESSION:"
        impression_split = re.split(r'IMPRESSION\s*[:.]?\s*', findings_split[1], maxsplit=1, flags=re.IGNORECASE)
        
        # If "IMPRESSION:" exists after "FINDINGS:"
        if len(impression_split) > 1:
            findings_text = impression_split[0].strip()
            # impression_match = re.search(r'.*?(?=\b[a-zA-Z]+:\b)', impression_split[1], re.IGNORECASE | re.DOTALL)
            # if impression_match:
            #     impression_text = impression_match.group().strip()
            # else:
            impression_text = impression_split[1].strip()
        else:
            # If there's no "IMPRESSION:", then all the content after "FINDINGS:" is considered as findings_text
            findings_text = impression_split[0].strip()

    # If "FINDINGS:" doesn't exist, but "IMPRESSION:" does, capture the impression_text
    else:
        impression_split = re.split(r'IMPRESSION\s*[:.]?\s*', text, maxsplit=1, flags=re.IGNORECASE)
        if len(impression_split) > 1:
            # impression_match = re.search(r'.*?(?=\b[a-zA-Z]+:\b)', impression_split[1], re.IGNORECASE | re.DOTALL)
            # if impression_match:
            #     impression_text = impression_match.group().strip()
            # else:
            impression_text = impression_split[1].strip()

    return findings_text, impression_text


def remove_phrases(text):
    phrases_to_remove = [
        "clinical indication:",
        "clinical information:",
        "EXAMINATION:",
        "clinical history:"
    ]

    # Find all occurrences of phrases to remove along with their start and end indices
    found_phrases = []
    for phrase in phrases_to_remove:
        for match in re.finditer(re.escape(phrase), text, re.IGNORECASE):
            # Search for the next word followed by ':' or 'comparison'
            end_match = re.search(r'\w+:\s|comparison', text[match.end():], re.IGNORECASE)
            if end_match:
                end_index = match.end() + end_match.start()
            else:
                end_index = len(text)
            found_phrases.append((match.start(), end_index))

    # Sort found phrases by their start index
    found_phrases.sort(key=lambda x: x[0])

    # Remove phrases from the text in reverse order to avoid messing up the indices
    for start_index, end_index in reversed(found_phrases):
        text = text[:start_index] + text[end_index:]

    return text


# Step3: find conditional sentences
def contains_conditional_keywords(sent, conditional_keywords):
    """Checks if the sentence contains any of the conditional keywords."""
    return any(word.text.lower() in conditional_keywords for word in sent)


# Step4: Find the ASPECT SCORE for the imaging
def get_aspect_score_values(text):
    scores = defaultdict(list)

    matches = re.findall("(right|left)?\s*aspect\s+score[\s=:]*(\d+)", text, re.IGNORECASE)    
    
    for side, score in matches:
        if side:
            scores[side.lower()].append(int(score))
        else:
            scores['general'].append(int(score))

    return scores


def clean_duplicate_words(text):
    return re.sub(r'\b(\w+\s*)\1{1,}', '\\1', text)


def custom_tokenizer(nlp):
    infix_re = re.compile(r'''[-.\,\?\:\;\/\...\‘\’\`\“\”\"\'~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=None)


def add_pathologies(nlp, ruler, PATHOLOGIES):
    PATHOLOGIES_BASE = [nlp(word)[0].lemma_ for word in PATHOLOGIES]
    
    ruler.add_patterns([
        {"label": "PATHOLOGY", "pattern": [
            {"DEP": {"IN": ["nmod", "nsubj", "attr"]}, "OP": "?"}, 
            {"LEMMA": {"IN": PATHOLOGIES_BASE}},
        ]},
        {"label": "PATHOLOGY", "pattern":[
            {"DEP": {"IN": ["nmod", "nsubj", "attr"]}, "OP": "?"},
            {"LOWER": "cerebrovascular"},
            {"LOWER": "accident"}, 
        ]},
        {"label": "PATHOLOGY", "pattern": [
            {"LOWER": {"IN": ["loss", "blurring", "reduced", "reduce"]}, "OP": "?"}, 
            {"LOWER": "of", "OP": "?"}, 
            {"LOWER": {"IN": ["gray", "grey"]}}, 
            {"ORTH": "-", "OP": "?"}, 
            {"LOWER": "white"},
        ]},
        {"label": "PATHOLOGY", "pattern": [
            {"LOWER": "blood"}, 
            {"LEMMA": "product"},
        ]},
        {"label":"PATHOLOGY", "pattern":[
            {"LOWER": "hyperdense"},
            {"LOWER": "material"},
        ]},
        {"label": "PATHOLOGY_PERFUSION", "pattern": [
            {"LOWER": {"IN": ["decreased", "decrease", "reduced", "reduce", "reduction"]}},
            {"LOWER": "in", "OP": "?"},
            {"DEP": {"IN": ["amod", "nmod", "nsubj", "attr"]}, "OP": "?"},
            {"LOWER": "blood", "OP": "?"},
            {"LOWER": "in", "OP": "?"},
            {"LOWER": "flow"},
        ]},
        {"label": "PATHOLOGY_PERFUSION", "pattern": [
            {"LOWER": {"IN": ["larger", "elevate", "increased", "increase", "prolonged", "prolong", "prolongation", "elevated"]}},
            {"LOWER": "surrounding", "OP":"?"},
            {"LEMMA": {"IN": ["area"]}, "OP":"?"},
            {"LOWER": "of", "OP":"?"},
            {"LOWER": "the", "OP":"?"},
            {"LOWER": {"IN": ["mean", "median"]}, "OP": "?"},
            {"LOWER": "transit", "OP": "?"},
            {"LOWER": {"IN": ["time", "tmax"]}},
            {"LOWER": "to", "OP": "?"},
            {"LOWER": "maximum", "OP": "?"},
            {"LOWER": {"IN":["perfusion", "peak"]}, "OP":"?"}
        ]},
        {"label": "PATHOLOGY_PERFUSION", "pattern": [
            {"LOWER": {"IN": ["larger", "increased", "increase", "prolong", "prolonged", "elevate", "elevated"]}},
            {"LOWER": {"IN":["tmax", "ttp", "t"]}},
            {"LOWER": "max", "OP":"?"},
        ]},
        {"label": "PATHOLOGY_PERFUSION", "pattern": [
            {"LOWER": {"IN":["delayed", "delay", "slow", "slower", "decreased", "decrease", "reduced", "reduce", "reduction"]}},
            {"LOWER": "in", "OP": "?"},
            {"DEP": {"IN": ["amod", "nmod", "nsubj", "attr"]}, "OP": "?"},
            {"LOWER": {"IN": ["perfusion", "rcbv"]}},
        ]},
        {"label": "PATHOLOGY2", "pattern": [
            {"LOWER": {"IN": ["large", "huge", "significant"]}, "OP":"?"},
            {"LOWER": {"IN": ["petechial", "parenchymal", "intraparenchymal"]}},
            {"LEMMA": {"IN": ["hemorrhage", "hemorrhages", "hemorrhagic", "blood"]}}
        ]},
        {"label": "PATHOLOGY2", "pattern": [
            {"LOWER": "subarachnoid"},
            {"LOWER": {"IN": ["blood", "hemorrhage"]}}
        ]},
        {"label": "PATHOLOGY2", "pattern": [ 
            {"LOWER": {"IN": ["worsen", "worsening","decreased", "decrease", "acute", "recent", "new", "subacute", "early"]}}, 
            {"DEP": {"IN": ["amod", "nmod", "nsubj", "attr"]}, "OP": "?"},
            {"LOWER": "attenuation"},
        ]},
        {"label": "PATHOLOGY2", "pattern":[
            {"LOWER": {"IN": ["sulcal", "sulci"]}},
            {"LOWER": {"IN": ["effacement", "effaced", "hyperdensities", "hyperdensity"]}},
        ]},
        {"label": "PATHOLOGY2", "pattern":[
            {"LOWER": {"IN": ["effacement", "effaced", "hyperdensities", "hyperdensity"]}},
            {"LOWER": "of", "OP": "?"},
            {"LOWER": "the", "OP": "?"},
            {"LOWER": {"IN": ["sulcal", "sulci"]}},
        ]},
        {"label": "PATHOLOGY2", "pattern":[
            {"LOWER": {"IN": ["worsening", "increased", "increase", "mild", "significant", "acute", "recent", "new", "subacute", "early", "partial", "complete"]}},
            {"DEP": {"IN": ["amod", "nmod", "nsubj", "attr"]}, "OP": "?"},
            {"LOWER": {"IN": ["effacement", "effaced"]}},
        ]},
        {"label": "PATHOLOGY2", "pattern": [
            {"LOWER": {"IN": ["worsening", "increased", "increase", "mild", "significant", "acute", "recent", "new", "subacute", "early"]}},
            {"DEP": {"IN": ["amod", "nmod", "nsubj", "attr", "compound"]}, "OP": "?"},
            {"LOWER": "mass"},
            {"LOWER": "effect"},
            {"LOWER": "with", "OP": "?"},
            {"LOWER": "effacement", "OP": "?"},
        ]},
        {"label": "PATHOLOGY2", "pattern": [
            {"LOWER": "mass"},
            {"LOWER": "effect"},
            {"LEMMA": "be"},
            {"LEMMA": {"IN": ["worsened", "increased", "increase", "mild", "significant", "acute", "recent", "new", "subacute", "early"]}},
        ]},
        # {"label": "PATHOLOGY2", "pattern":[
        #     {"LOWER": "midline"},
        #     {"LOWER": "shift"},
        # ]},
        # {"label": "PATHOLOGY2", "pattern":[
        #     {"LOWER": "shift"},
        #     {"LOWER": "of", "OP": "?"},
        #     {"LOWER": "midline"}
        # ]},
        {"label": "PATHOLOGY2", "pattern": [
            {"LOWER": {"IN": ["restricted", "restriction"]}},
            {"DEP": {"IN": ["amod", "nmod", "nsubj", "attr"]}, "OP": "?"}, 
            {"LEMMA": {"IN": ["diffusion"]}}, 
        ]},
        # {"label": "CTAPATHOLOGY", "pattern": [
        #     # {"LEMMA": {"IN": ["be", "have"]}, "POS": "AUX", "OP": "?"},
        #     {"DEP": {"IN": ["nmod", "nsubj", "attr"]}, "OP": "?"}, 
        #     {"LEMMA": {"IN": ["stenosis", "occlusion"]}},
        #     # {"POS": "VERB", "TAG": {"IN": ["VBZ", "VBP"]}}
        # ]},
    ])
    ts = termset("en_clinical_sensitive")
    ts.add_patterns({"pseudo_negations": ["was not present"],
                     "preceding_negations": ["nonspecific", "not developed", "without expansion", "no expansion", "no demarcation", "no progression", "without progressive", "minimally increased", "slightly increased", "not increased", "minimal progression", "minimal expansion", "micro", "tiny", "insignificant", "without significant", "not significant", "without interval", "without evidence", "discordant", "do not cause", "no evidence of", "there is no", "there is no evidence of"], 
                     "following_negations": ["no", "nonspecific", "not visualized on this", "not developed", "without expansion", "no expansion", "no demarcation", "no progression", "without progressive", "minimally increased", "slightly increased", "not increased", "minimal progression", "minimal expansion", "without interval", "without evidence", "improved", "preserved", "micro", "tiny", "none", "there is no", "insignificant", "without significant", "not significant", "no evidence of", "there is no evidence of"]})
    nlp.add_pipe("negex", last=True, config={
        "ent_types": ["PATHOLOGY", "PATHOLOGY_PERFUSION", "PATHOLOGY2", "CTAPATHOLOGY", "WORSE_DESC"],
        "neg_termset": ts.get_patterns(),
    })


def _children_with_entity_getter(entity):
    def _getter(span):
        sentence = span.sent
        return [
            ent
            for ent in sentence.ents
            if ent.label_ == entity
        ]
    return _getter



def add_pathology_extensions(ruler, ACUTE_DESCS, NON_ACUTE_DESCS, WORSE_DESCS, REGIONS, CERTAINTIES, DENSITIES):
    patterns = [
        {"label": "ACUTE_DESC", "pattern": [{"LOWER": {"IN": ACUTE_DESCS}}]},#"POS": {"IN": ["pcomp", "noun", "pobj", "dobj", "amod", "nmod", "nsubj", "attr", "compound"]}}]},
        {"label": "ACUTE_DESC", "pattern": [{"LOWER": "age", "OP": "?"}, {"ORTH": "-", "OP": "?"}, {"LOWER": {"IN": ["indeterminate"]}}]},
        {"label": "NON_ACUTE_DESC", "pattern": [{"LOWER": {"IN": NON_ACUTE_DESCS}}]},
        {"label": "WORSE_DESC", "pattern": [{"LOWER": {"IN": WORSE_DESCS}}]},
        {"label": "WORSE_DESC", "pattern": [{"LOWER": "significantly"}, {"LOWER": "more"}]},
        {"label": "HISTORICAL", "pattern":[{"LOWER": {"IN":["recent", "earlier", "early"]}}, {"LOWER": {"IN": ["mri", "ct", "perfusion", "scan", "image"]}}]},
        {"label": "HISTORICAL", "pattern": [{"LOWER": {"IN": ["unchanged", "evolution", "evolving", "evolved", "evolve", "prior", "compared", "comparison", "previous", "previously", "redemonstration", "again", "since", "presumed", "continued", "known", "stable", "established"]}}]},
        {"label": "REGION", "pattern": [{"LOWER": {"IN": REGIONS}}]},
        {"label": "CERTAINTY", "pattern": [{"LOWER": "less", "OP": "?"}, {"LOWER": {"IN": CERTAINTIES}}]},
        {"label": "DENSITY", "pattern": [{"LOWER": "slightly", "OP": "?"}, {"LOWER": {"IN": DENSITIES}}]},
    ]
    ruler.add_patterns(patterns)

    Span.set_extension("regions", getter=_children_with_entity_getter("REGION"))
    Span.set_extension("certainties", getter=_children_with_entity_getter("CERTAINTY"))
    Span.set_extension("acute_descs", getter=_children_with_entity_getter("ACUTE_DESC"))
    Span.set_extension("non_acute_descs", getter=_children_with_entity_getter("NON_ACUTE_DESC"))
    Span.set_extension("worse_descs", getter=_children_with_entity_getter("WORSE_DESC"))
    Span.set_extension("densities", getter=_children_with_entity_getter("DENSITY"))
    Span.set_extension("historical", getter=_children_with_entity_getter("HISTORICAL"))



class ReportLabeler:
    def __init__(self, certain_words, uncertain_words, keep_as_is_lists):
        self.certain_words = certain_words
        self.uncertain_words = uncertain_words
        self.keep_as_is_lists = keep_as_is_lists

        self.nlp = spacy.load("en_core_sci_sm")
        self.nlp.disable_pipe("ner")
        self.nlp.tokenizer = custom_tokenizer(self.nlp)
        ruler = self.nlp.add_pipe("entity_ruler")
        add_pathologies(self.nlp, ruler, PATHOLOGIES)
        add_pathology_extensions(ruler, ACUTE_DESCS, NON_ACUTE_DESCS, WORSE_DESCS, REGIONS, CERTAINTIES, DENSITIES)
    
    
    def label_report(self, report_text, exam_code):
        if pd.isna(report_text):
            return None, ["No report text"]
        report_text = re.sub(r"ACUTE MCA TERRITORY INVOLVED\s*:\S*", "", report_text)
        report_text = insert_space_after_period_percent(report_text)
        report_text = custom_segment(report_text, self.keep_as_is_lists)
        report_text = clean_duplicate_words(report_text)
        perfusion = is_perfusion(report_text)

        findings_text, impression_text = extract_findings_impression(report_text)

        if not pd.isna(impression_text):
            primary_text = impression_text
            secondary_text = remove_phrases(report_text) if pd.isna(findings_text) else findings_text
        elif not pd.isna(findings_text):
            primary_text = findings_text
            secondary_text = None
        else:
            primary_text = remove_phrases(report_text)
            secondary_text = None

        doc = self.nlp(primary_text)
        label, reasons = self._decide_label(doc, exam_code, perfusion)

        if label == StrokeLabel.POSITIVE:
            return label, reasons

        if secondary_text is None:
            return StrokeLabel.NEGATIVE, reasons

        doc = self.nlp(secondary_text)
        return self._decide_label(doc, exam_code, perfusion)

    def _decide_label(self, doc, exam_code, perfusion):
        CONDITOINAL_KEYWORDS = ["if", "when"]
        
        stroke_reasons = []
        no_stroke_reasons = []
        
        found_positive = False
        found_negative = False
        # displacy.serve(doc, style="dep", port=5001)
        for ent in doc.ents:
            # if ent.label_ in ['PATHOLOGY', 'PATHOLOGY2', 'PATHOLOGY_PERFUSION']:
            #     print()
            #     if ent._.negex:
            #         print("*NO* ", end='')
            #     print(ent.text)
            #     print("- region:", ent._.regions)
            #     print("- certainty:", ent._.certainties)
            #     print("- historical:", ent._.historical)
            #     print("- acute descs:", ent._.acute_descs)
            #     print("- non-acute descs:", ent._.non_acute_descs)
            #     print("- worse-descs:", ent._.worse_descs)
            #     print("- densities:", ent._.densities)
            #     print("- source:", ent.root.sent)
            # Ignore entities in conditional sentences
            if contains_conditional_keywords(ent.root.sent, CONDITOINAL_KEYWORDS):
                continue
            # If pathology and not negated and has acute descriptions
            if ent.label_ == 'PATHOLOGY' and not ent._.negex and (len(ent._.acute_descs) != 0 or len(ent._.worse_descs) != 0) and (len(ent._.historical) == 0 or len(ent._.worse_descs) != 0) and ((any(word.text.lower() in self.certain_words for word in ent._.certainties)) or len(ent._.certainties) == 0):
                found_positive = True
                stroke_reasons.append(ent.root.sent.text)
            elif ent.label_ == 'PATHOLOGY2' and not ent._.negex and len(ent._.non_acute_descs) == 0 and (len(ent._.historical) == 0 or len(ent._.worse_descs) != 0) and ((any(word.text.lower() in self.certain_words for word in ent._.certainties)) or len(ent._.certainties) == 0):
                found_positive = True
                stroke_reasons.append(ent.root.sent.text)
            elif ent.label_ == 'PATHOLOGY_PERFUSION' and (exam_code in ['CTNHBPER2', 'CTNHBPER1'] or perfusion) and not ent._.negex and len(ent._.non_acute_descs) == 0 and (len(ent._.historical) == 0 or len(ent._.worse_descs) != 0) and ((any(word.text.lower() in self.certain_words for word in ent._.certainties)) or len(ent._.certainties) == 0):
                found_positive = True
                stroke_reasons.append(ent.root.sent.text)
            # elif ent.label_ == 'CTAPATHOLOGY' and (exam_code in ['CANHEAD1', 'CANHEAD2', 'CANNECK1', 'CANNECK2'])and not ent._.negex and len(ent._.non_acute_descs) == 0 and len(ent._.historical) == 0 and ((any(word.text.lower() in self.certain_words for word in ent._.certainties)) or len(ent._.certainties) == 0):
            #     found_positive = True
            #     stroke_reasons.append(ent.root.sent.text)
            else:
                found_negative = True
                no_stroke_reasons.append(ent.root.sent.text)
        
        if side_scores := get_aspect_score_values(doc.text):
            for side, scores in side_scores.items():
                if any(score < 10 for score in scores):
                    found_positive = True
                    stroke_reasons.append(f"{side} aspect score is less than 10")
            
        # Making the final decision based on the results
        if found_positive:
            return StrokeLabel.POSITIVE, stroke_reasons
        elif found_negative:
            return StrokeLabel.NEGATIVE, no_stroke_reasons
        else:
            return StrokeLabel.NEGATIVE, ['Matching patterns were not found']


def label_reports(data, labeler):    
    labels = []
    reasons_joined = []

    for report_text, exam_code in tqdm(zip(data["ReportText"], data["ExamCode"]), desc="Processing reports"):
        label, reasons = labeler.label_report(report_text, exam_code)
        labels.append(label)
        reasons_joined.append('; '.join(reasons))

    data['Label'] = labels
    data['Reason'] = reasons_joined
    return data


def is_perfusion(report_text):
    return re.search('perfusion', report_text, re.IGNORECASE) is not None
    

class StrokeLabel(Enum):
    POSITIVE = 'STROKE'
    NEGATIVE = 'NO STROKE'


if __name__ == '__main__':
    # data = load_accession_data()
    data = pd.read_csv('data/imaging_excluding_angio.csv')

    CERTAIN_WORDS = ["likely", "probable", "possible", "suspected", "suspicious", "definitive"]
    UNCERTAIN_WORDS = ["unlikely", "uncertain", "less likely"]

    keep_as_is_lists = [PATHOLOGIES, ["hyperdensities", "hyperdensity", "sulcal", "intraparenchymal", "petechial", "subarachnoid", "stenosis", "occlusion", "attenuation", "diffusion", "indeterminate", "midline", "mri", "ct", "perfusion", "scan", "image", "prior", "compared", "comparison", "previous", "redemonstration", "again", "since", "persumed", "continued", "known", "stable", "established", "evolving"], REGIONS, ACUTE_DESCS, NON_ACUTE_DESCS, WORSE_DESCS, CERTAINTIES, DENSITIES]
    labeler = ReportLabeler(CERTAIN_WORDS, UNCERTAIN_WORDS, keep_as_is_lists)
    new_data = label_reports(data, labeler)
    new_data.to_csv('output_data/NeuroAccessionsLabeledv22AllAccessions.csv', index=False)
