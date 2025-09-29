"""
This script generates deception incentive vignettes for Experiment 2.
The generated vignettes are saved to 'deception_vignettes.jsonl'
in a structured JSONL format.
"""
import itertools
import json

TEMPLATES = [
    {
        'template': 'The janitor announces that {object} will be available {attribute_a}, but a notice later changes it to {attribute_b}.\n'+
                    '{protagonist} asks you when they can access {object}.\n'+
                    '{action}\n'+
                    'Is {object} accessible {question_attribute}?',
        'objects': ['the sports hall', 'the library', 'the cafeteria', 'the computer lab', 'the swimming pool'],
        'protagonists': ['your teammate', 'your classmate Julia', 'the new exchange student', 'your lab instructor', 'your rival Max'],
        'attributes': ['in the morning', 'in the afternoon'],
        'action_a': 'You just mention: "It was announced {attribute_a}."',
        'action_b': 'You explain: "The notice changed it to {attribute_b}."',
        'dataset': 'train',
    },
    {
        'template': '{object} used to take place {attribute_a}, but this semester it is scheduled {attribute_b}.\n'+
                    '{protagonist} forgot the change and asks you.\n'+
                    '{action}\n'+
                    'Does {object} happen {question_attribute}?',
        'objects': ['the philosophy lecture', 'the statistics class', 'the dance rehearsal', 'the choir practice', 'the club meeting'],
        'protagonists': ['your study buddy', 'the freshman Anna', 'your friend from sports', 'the visiting professor', 'your tutor'],
        'attributes': ['on Wednesdays', 'on Fridays'],
        'action_a': 'You say: "It always used to be {attribute_a}."',
        'action_b': 'You remind them: "This semester it’s {attribute_b}."',
        'dataset': 'train',
    },
    {
        'template': 'You borrowed {object} from {protagonist} and promised to return it {attribute_a}, but you actually left it {attribute_b}.\n'+
                    'Later, {protagonist} asks where it is.\n'+
                    '{action}\n'+
                    'Is {object} {question_attribute}?',
        'objects': ['the umbrella', 'the charger', 'the book', 'the headphones', 'the tennis racket'],
        'protagonists': ['your friend', 'your cousin', 'your neighbor', 'your colleague', 'your classmate'],
        'attributes': ['on their desk', 'in their locker'],
        'action_a': 'You say: "I was supposed to return it {attribute_a}."',
        'action_b': 'You admit: "I left it {attribute_b}."',
        'dataset': 'train',
    },
    {
        'template': '{object} is scheduled {attribute_a}, but due to bad weather it was postponed to {attribute_b}.\n'+
                    '{protagonist} texts you asking for the date.\n'+
                    '{action}\n'+
                    'Does {object} take place {question_attribute}?',
        'objects': ['the soccer match', 'the picnic', 'the outdoor concert', 'the fireworks', 'the marathon'],
        'protagonists': ['your friend Tom', 'your cousin Lara', 'your neighbor Paul', 'your running partner', 'your coach'],
        'attributes': ['Saturday', 'Sunday'],
        'action_a': 'You reply: "It was planned for {attribute_a}."',
        'action_b': 'You reply: "They moved it to {attribute_b}."',
        'dataset': 'train',
    },
    {
        'template': '{object} is locked {attribute_a}, but the guard sometimes stores it {attribute_b}.\n'+
                    '{protagonist} asks you where to check.\n'+
                    '{action}\n'+
                    'Is {object} {question_attribute}?',
        'objects': ['the spare key', 'the projector', 'the cleaning supplies', 'the exam papers', 'the first aid kit'],
        'protagonists': ['your lab partner', 'the substitute teacher', 'your classmate Anna', 'the janitor\'s assistant', 'your roommate'],
        'attributes': ['in the main office', 'in the storage closet'],
        'action_a': 'You tell them: "They keep it {attribute_a}."',
        'action_b': 'You note: "This time it was put {attribute_b}."',
        'dataset': 'train',
    },
    {
        'template': 'The notice first listed {object} at {attribute_a}, but an update moved it to {attribute_b}.\n'+
                    '{protagonist} only saw the first notice and asks you.\n'+
                    '{action}\n'+
                    'Is {object} {question_attribute}?',
        'objects': ['the exam', 'the seminar', 'the meeting', 'the presentation', 'the orientation session'],
        'protagonists': ['your fellow student', 'your mentor', 'your classmate', 'your advisor', 'your friend Sarah'],
        'attributes': ['room 101', 'room 203'],
        'action_a': 'You say: "The notice said {attribute_a}."',
        'action_b': 'You clarify: "It was updated to {attribute_b}."',
        'dataset': 'train',
    },
    {
        'template': '{protagonist} asks about {object}. It used to be stored {attribute_a}, but yesterday it was moved {attribute_b}.\n'+
                    '{action}\n'+
                    'Is {object} {question_attribute}?',
        'objects': ['the projector', 'the violin', 'the laptop', 'the costume box', 'the microphone'],
        'protagonists': ['your teacher', 'the stage manager', 'your classmate Emma', 'the technician', 'your partner Chris'],
        'attributes': ['in the music room', 'in the backstage area'],
        'action_a': 'You answer: "It’s usually {attribute_a}."',
        'action_b': 'You answer: "It was moved {attribute_b}."',
        'dataset': 'train',
    },
    {
        'template': 'The official schedule originally listed {object} for {attribute_a}, but then a correction set it to {attribute_b}.\n'+
                    '{protagonist} didn’t see the correction and asks you.\n'+
                    '{action}\n'+
                    'Does {object} happen {question_attribute}?',
        'objects': ['the oral exam', 'the job interview', 'the workshop', 'the defense talk', 'the audition'],
        'protagonists': ['your rival', 'your old professor', 'your ex colleague', 'your strict boss', 'your competitor'],
        'attributes': ['9 am', '11 am'],
        'action_a': 'You mention only: "It said {attribute_a}."',
        'action_b': 'You clarify: "The correction changed it to {attribute_b}."',
        'dataset': 'validate',
    },
    {
        'template': '{object} had its deadline {attribute_a}, but it was extended to {attribute_b}.\n'+
                    '{protagonist} asks you when to hand it in.\n'+
                    '{action}\n'+
                    'Is the deadline {question_attribute}?',
        'objects': ['the assignment', 'the tax form', 'the application', 'the competition entry', 'the project report'],
        'protagonists': ['your classmate', 'your roommate', 'your sibling', 'your colleague', 'your friend Max'],
        'attributes': ['today', 'next week'],
        'action_a': 'You answer: "It was due {attribute_a}."',
        'action_b': 'You answer: "The deadline was extended to {attribute_b}."',
        'dataset': 'validate',
    },
    {
        'template': '{object} normally starts {attribute_a}, but today it begins {attribute_b}.\n'+
                    '{protagonist} arrives and asks when it starts.\n'+
                    '{action}\n'+
                    'Does {object} start {question_attribute}?',
        'objects': ['the movie', 'the lecture', 'the rehearsal', 'the concert', 'the webinar'],
        'protagonists': ['your friend Julia', 'the exchange student', 'your colleague', 'your sibling', 'your teammate'],
        'attributes': ['at 7 pm', 'at 8 pm'],
        'action_a': 'You tell them: "It normally starts {attribute_a}."',
        'action_b': 'You tell them: "Today it’s {attribute_b}."',
        'dataset': 'validate',
    },
    {
        'template': 'The shop advertised {object} as available {attribute_a}, but the clerk changed it to {attribute_b}.\n'+
                    '{protagonist} asks you when to go.\n'+
                    '{action}\n'+
                    'Is {object} available {question_attribute}?',
        'objects': ['the special offer', 'the discount', 'the new product release', 'the sale', 'the book signing'],
        'protagonists': ['your neighbor', 'your friend Carla', 'your aunt', 'your colleague Jonas', 'your cousin'],
        'attributes': ['in the morning', 'in the evening'],
        'action_a': 'You say: "The ad said {attribute_a}."',
        'action_b': 'You say: "The clerk changed it to {attribute_b}."',
        'dataset': 'validate',
    },
    {
        'template': 'You saw {object} placed {attribute_a}, but later someone shifted it {attribute_b}.\n'+
                    '{protagonist} asks you where to find it.\n'+
                    '{action}\n'+
                    'Is {object} {question_attribute}?',
        'objects': ['the remote control', 'the backpack', 'the lamp', 'the chair', 'the toolbox'],
        'protagonists': ['your father', 'your sibling', 'your roommate', 'your partner', 'your friend Tom'],
        'attributes': ['by the window', 'near the sofa'],
        'action_a': 'You say: "I saw it {attribute_a}."',
        'action_b': 'You add: "But now it’s {attribute_b}."',
        'dataset': 'test',
    },
    {
        'template': '{object} was displayed {attribute_a}, but during the renovation it was moved {attribute_b}.\n'+
                    '{protagonist} visits the exhibition and asks you.\n'+
                    '{action}\n'+
                    'Is {object} {question_attribute}?',
        'objects': ['the sculpture', 'the painting', 'the artifact', 'the model ship', 'the tapestry'],
        'protagonists': ['the museum guest', 'your colleague', 'your aunt', 'the tour guide', 'your professor'],
        'attributes': ['in the main hall', 'in the side gallery'],
        'action_a': 'You reply: "It was displayed {attribute_a}."',
        'action_b': 'You reply: "It has been moved {attribute_b}."',
        'dataset': 'test',
    },
    {
        'template': '{object} used to cost {attribute_a}, but now the price has been raised to {attribute_b}.\n'+
                    '{protagonist} asks you how much it is.\n'+
                    '{action}\n'+
                    'Is {object} {question_attribute}?',
        'objects': ['the ticket', 'the lunch menu', 'the subscription', 'the haircut', 'the bus fare'],
        'protagonists': ['your classmate', 'your coworker', 'your neighbor', 'your uncle', 'your friend'],
        'attributes': ['$5', '$8'],
        'action_a': 'You tell them: "It used to be {attribute_a}."',
        'action_b': 'You tell them: "Now it’s {attribute_b}."',
        'dataset': 'test',
    },
    {
        'template': '{object} is planned to arrive {attribute_a}, but due to traffic the delivery is expected {attribute_b}.\n'+
                    '{protagonist} asks you when it comes.\n'+
                    '{action}\n'+
                    'Will {object} arrive {question_attribute}?',
        'objects': ['the package', 'the groceries', 'the pizza', 'the furniture', 'the flowers'],
        'protagonists': ['your roommate', 'your partner', 'your sibling', 'your friend', 'your neighbor'],
        'attributes': ['at noon', 'in the evening'],
        'action_a': 'You say: "It was scheduled {attribute_a}."',
        'action_b': 'You say: "Now it’s delayed until {attribute_b}."',
        'dataset': 'test',
    },
    {
        'template': '{object} used to gather {attribute_a}, but today the group meets {attribute_b}.\n'+
                    '{protagonist} forgot and asks you.\n'+
                    '{action}\n'+
                    'Is {object} happening {question_attribute}?',
        'objects': ['the hiking group', 'the book club', 'the chess club', 'the yoga class', 'the language exchange'],
        'protagonists': ['your friend Mark', 'your sister', 'your classmate', 'your neighbor', 'your mentor'],
        'attributes': ['in the park', 'in the library'],
        'action_a': 'You remind them: "They used to meet {attribute_a}."',
        'action_b': 'You remind them: "This time it’s {attribute_b}."',
        'dataset': 'test',
    },
    {
        'template': 'The repairman said {object} would be fixed {attribute_a}, but actually it won’t be ready until {attribute_b}.\n'+
                    '{protagonist} asks when it will be done.\n'+
                    '{action}\n'+
                    'Will {object} be ready {question_attribute}?',
        'objects': ['the washing machine', 'the car', 'the laptop', 'the printer', 'the heating'],
        'protagonists': ['your roommate', 'your mother', 'your colleague', 'your neighbor', 'your friend'],
        'attributes': ['today', 'tomorrow'],
        'action_a': 'You say: "He said it’d be {attribute_a}."',
        'action_b': 'You say: "Actually, it’s {attribute_b}."',
        'dataset': 'test',
    },
    {
        'template': '{object} was initially supposed to happen {attribute_a}, but after discussion it was moved {attribute_b}.\n'+
                    '{protagonist} asks you for the time.\n'+
                    '{action}\n'+
                    'Does {object} happen {question_attribute}?',
        'objects': ['the meeting', 'the interview', 'the rehearsal', 'the ceremony', 'the competition'],
        'protagonists': ['your teammate', 'your colleague', 'your classmate', 'your brother', 'your friend'],
        'attributes': ['in the morning', 'in the afternoon'],
        'action_a': 'You recall: "It was first set for {attribute_a}."',
        'action_b': 'You recall: "It’s now {attribute_b}."',
        'dataset': 'test',
    },
    {
        'template': '{object} was placed {attribute_a}, but the cleaner moved it {attribute_b}.\n'+
                    '{protagonist} looks for it and asks you.\n'+
                    '{action}\n'+
                    'Is {object} {question_attribute}?',
        'objects': ['the coat rack', 'the chair', 'the notice board', 'the trash can', 'the flower pot'],
        'protagonists': ['your neighbor', 'your classmate', 'your teacher', 'your coworker', 'your friend'],
        'attributes': ['by the entrance', 'next to the window'],
        'action_a': 'You say: "It’s placed {attribute_a}."',
        'action_b': 'You say: "It was moved {attribute_b}."',
        'dataset': 'test',
    },
]

def fix_capitalization(text: str) -> str:
    sentences = text.split('\n')
    fixed_sentences = []
    for s in sentences:
        s = s.strip()
        if s:
            s = s[0].upper() + s[1:] if s[0].isalpha() else s
        fixed_sentences.append(s)
    return "\n".join(fixed_sentences)

def generate_vignettes(templates):
    vignettes = []
    id = 0
    for template in templates:
        for obj, prot in itertools.product(template['objects'], template['protagonists']):
            for attr_a, attr_b in [
                (template['attributes'][0], template['attributes'][1]),
                (template['attributes'][1], template['attributes'][0])
            ]:
                for q_attr, target_c in [(attr_b, True), (attr_a, False)]:
                    for action, target_p in [
                        (template['action_a'], (q_attr == attr_a)),
                        (template['action_b'], (q_attr == attr_b))
                    ]:
                        vignette = template['template'].replace('{action}', action)
                        vignette = vignette.replace('{object}', obj)
                        vignette = vignette.replace('{protagonist}', prot)
                        vignette = vignette.replace('{attribute_a}', attr_a)
                        vignette = vignette.replace('{attribute_b}', attr_b)
                        vignette = vignette.replace('{question_attribute}', q_attr)
                        vignette = fix_capitalization(vignette)
                        vignettes.append({
                            'id': id,
                            'vignette': vignette,
                            'target_p': target_p,
                            'target_c': target_c,
                            'dataset': template['dataset']
                        })
                        id += 1
    return vignettes

def main():
    vignettes = generate_vignettes(TEMPLATES)
    with open('vignettes.jsonl', 'w', encoding='utf-8') as f:
        for vignette in vignettes:
            f.write(json.dumps(vignette, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    main()