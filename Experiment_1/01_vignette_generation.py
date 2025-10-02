"""
This script generates belief inference vignettes for Experiment 1.
The generated vignettes are saved to 'generated_vignettes.jsonl'
in a structured JSONL format.
"""
import itertools
import json
SPLITNAMES = {'train': 'train', 'validate': 'validate', 'test': 'test'}

TEMPLATES = [
    {
        'template': '{object} is usually {attribute_a}, but earlier today you took it away and placed it {attribute_b}.\n'+
                    '{protagonist} asks you where {object} is.\n'+
                    '{action}\n'+
                    'Is {object} {question_attribute}?',
        'objects': ['the key', 'the wallet', 'the headphones'],
        'protagonists': ['your three year old brother Bob', 'your guest Dave', 'your grandmother'],
        'attributes': ['in the kitchen drawer', 'on the living room shelf'],
        'action_a': 'You answer: "It\'s usually {attribute_a}."',
        'action_b': 'You answer: "I placed it {attribute_b}."',
        'dataset': SPLITNAMES['train'],
        'template_id': 0,
    },
    {
        'template': 'You invited {protagonist} for your birthday on {attribute_b}.\n'+
                    'Now, you receive a message that {object} is rescheduled from {attribute_a} to {attribute_b}.\n'+
                    'You know that {protagonist} plans to go to {object}, but that they didn\'t receive this message.\n'+
                    '{action}\n'+
                    'Does {object} take place on {question_attribute}?',
        'objects': ['the local football match', 'the firework at the fair', 'the town hall meeting'],
        'protagonists': ['your neighbor Greg', 'your colleague Kirk', 'your friend Hannah'],
        'attributes': ['Friday this week', 'Friday next week'],
        'action_a': 'You delete the message from your inbox.',
        'action_b': 'You forward the message to {protagonist}.',
        'dataset': SPLITNAMES['train'],
        'template_id': 1,
    },
    {
        'template': 'The janitor just moved {object} from its place {attribute_a} to {attribute_b}.\n'+
                    '{protagonist} comes looking for {object} and asks you where it is.\n'+
                    '{action}\n'+
                    'Is {object} {question_attribute}?',
        'objects': ['the lost-and-found box', 'the recycling bin', 'the notice board'],
        'protagonists': ['the principal', 'your teacher', 'the new kid'],
        'attributes': ['in the entrance hall', 'next to the gym'],
        'action_a': 'You answer: "Search for it {attribute_a}."',
        'action_b': 'You answer: "It was moved {attribute_b}."',
        'dataset': SPLITNAMES['train'],
        'template_id': 2,
    },
    {
        'template': '{protagonist} enters the lecture hall late and asks you where to find {object}.\n'+
                    'That\'s because {object} often sits in {attribute_a} of the lecture hall, but when you entered you saw them sit in {attribute_b}.\n'+
                    'You tell them: "Look over there", pointing to {action}.\n'+
                    'Is {object} sitting in {question_attribute}?',
        'objects': ['the student assistant', 'the dean', 'the debate team captain'],
        'protagonists': ['your lab partner Leo', 'your shy friend Maria', 'your classmate Nina'],
        'attributes': ['the front row', 'the back row'],
        'action_a': '{attribute_a}',
        'action_b': '{attribute_b}',
        'dataset': SPLITNAMES['train'],
        'template_id': 3,
    },
    {
        'template': 'The librarian reshelves {object}, which you want to use for your research, from {attribute_a} to {attribute_b}.\n'+
                    '{protagonist} wants to borrow {object} and asks you where to find it.\n'+
                    '{action}\n'+
                    'Is {object} located in {question_attribute}?',
        'objects': ['the encyclopedia', 'the math textbook', 'the atlas'],
        'protagonists': ['your study partner Sam', 'another student from your class', 'a random stranger'],
        'attributes': ['the reference section', 'the non-fiction shelf'],
        'action_a': 'You say: "Earlier today, I saw it in {attribute_a}."',
        'action_b': 'You say: "It has been reshelved to {attribute_b}."',
        'dataset': SPLITNAMES['validate'],
        'template_id': 4,
    },
    {
        'template': 'You normally keep {object} {attribute_a}, but you needed it earlier today and put it {attribute_b}.\n'+
                    '{protagonist} needs {object} and asks you where it is.\n'+
                    '{action}\n'+
                    'Is {object} {question_attribute}?',
        'objects': ['the Aspirin', 'the cough syrup', 'the Ibuprofen'],
        'protagonists': ['your drunk father', 'your sick grandmother', 'your shady roommate'],
        'attributes': ['in the bathroom cabinet', 'in the kitchen drawer'],
        'action_a': 'You reply: "Why wouldn\'t it be {attribute_a}?"',
        'action_b': 'You reply: "I put it {attribute_b}."',
        'dataset': SPLITNAMES['validate'],
        'template_id': 5,
    },
    {
        'template': '{protagonist} is notoriously unreliable.\n'+
                    'Some minutes ago you parked {object} {attribute_b}. You usually park {object} {attribute_a}.\n'+
                    'Now {protagonist} wants to borrow {object} for some hours and asks you where it is.\n'+
                    '{action}\n'+
                    'Is {object} {question_attribute}?',
        'objects': ['the car', 'the motorbike', 'the bicycle'],
        'protagonists': ['your friend Quentin', 'your neighbor Steve', 'your sibling Uma'],
        'attributes': ['in the garage', 'on the street'],
        'action_a': 'You answer: "You know that I always park it {attribute_a}."',
        'action_b': 'You answer: "Today I left it {attribute_b}."',
        'dataset': SPLITNAMES['validate'],
        'template_id': 6,
    },
    {
        'template': 'Due to a delay, {object} is now {attribute_b}, even though it was originally planned for {attribute_a}.\n'+
                    '{protagonist} plans to take {object} and asks you when it is. You remember how {protagonist} once treated you badly.\n'+
                    '{action}\n'+
                    'Does {object} take place on {question_attribute}?',
        'objects': ['the flight', 'the train', 'the bus ride'],
        'protagonists': ['your colleague Alex', 'your boss Clara', 'your acquantaince Emma'],
        'attributes': ['Monday morning', 'Tuesday afternoon'],
        'action_a': 'You show them the original schedule.',
        'action_b': 'You show them the delay notification.',
        'dataset': SPLITNAMES['validate'],
        'template_id': 7,
    },
    {
        'template': '{protagonist} tries to hit on {object} for quite a while now.\n'+
                    '{object} would love to start {attribute_b}, she got bored of {attribute_a}, which she does often.\n'+
                    'Now {protagonist} wants make {object} a present for her birthday and asks you what to get.\n'+
                    '{action}\n'+
                    'Would {object} like a present for {question_attribute}?',
        'objects': ['your crush Linda', 'your sister Hannah', 'your teammate Zoe'],
        'protagonists': ['your friend Victor', 'your classmate Paul', 'that jerk rick'],
        'attributes': ['reading books', 'playing computer games'],
        'action_a': 'You suggest that, since she\'s always {attribute_a}, she might like a new one of those.',
        'action_b': 'You tell them that even though she\'s {attribute_a} often, she would love to start {attribute_b}."',
        'dataset': SPLITNAMES['test'],
        'template_id': 8,
    },
    {
        'template': '{protagonist} enters and tells you: "Oh Lord, am I longing for {object}! I\'ll take some from {attribute_a} if you don\'t mind".\n'+
                    'Just a moment ago you hid them in {attribute_b} so no one would take it.\n'+
                    '{action}\n'+
                    'Are {object} in {question_attribute}?',
        'objects': ['the cookies', 'the snacks', 'the chips'],
        'protagonists': ['your roommate George', 'your sibling Hannah', 'your father'],
        'attributes': ['the fridge', 'the cupboard'],
        'action_a': 'You just shrug and say "Sure, go ahead."',
        'action_b': 'You point to {attribute_b} and tell them to look there.',
        'dataset': SPLITNAMES['test'],
        'template_id': 9,
    },
    {
        'template': 'It\'s open house day at your school. While {object} normally wear {attribute_a}, for this occasion they wear {attribute_b}.\n'+
                    '{protagonist} wants to join {object} and asks you what to wear for the taster session that they offer.\n'+
                    '{action}\n'+
                    'Are {object} wearing {question_attribute}?',
        'objects': ['the debate club members', 'the environmental volunteers', 'the school choir members'],
        'protagonists': ['your acquaintance Jeff', 'your annoying cousin Larry', 'the classmate you can\'t stand'],
        'attributes': ['all black', 'blue shorts and a white t-shirt'],
        'action_a': 'You tell them: "They normally wear {attribute_a}."',
        'action_b': 'You tell them: "On open house day, they are wearing {attribute_b}."',
        'dataset': SPLITNAMES['test'],
        'template_id': 10,
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
                            'dataset': template['dataset'],
                            'template_id': template['template_id'],
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