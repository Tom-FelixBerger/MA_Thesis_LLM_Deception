"""
This script generates belief inference vignettes for Experiment 1.
The generated vignettes are saved to 'generated_vignettes.jsonl'
in a structured JSONL format.
"""
import itertools
import json

TEMPLATES = [
    {
        'template': '{object} is usually {attribute_a}, but earlier today you took it away and placed it {attribute_b}.\n'+
                    '{protagonist} asks you where {object} is.\n'+
                    '{action}\n'+
                    'Is {object} {question_attribute}?',
        'objects': ['the key', 'the wallet', 'the ming vase', 'the painting', 'the computer'],
        'protagonists': ['your friend Alice', 'your brother Bob', 'the burglar Charlie', 'your guest Dave', 'your coworker Eve'],
        'attributes': ['in the kitchen drawer', 'on the living room shelf'],
        'action_a': 'You answer: "It\'s usually {attribute_a}."',
        'action_b': 'You answer: "I placed it {attribute_b}."',
        'dataset': 'train',
    },
    {
        'template': 'You receive a message that {object} is rescheduled from {attribute_a} to {attribute_b}, but that\'s when you also celebrate your birthday.\n'+
                    '{protagonist} is invited to your birthday, but also plans to go to {object}.\n'+
                    '{action}\n'+
                    'Does {object} take place on {question_attribute}?',
        'objects': ['the match', 'the appointment', 'the dinner', 'the show', 'the meeting'],
        'protagonists': ['your neighbor Greg', 'your colleague Kirk', 'your mother', 'your ex partner', 'your friend Hannah'],
        'attributes': ['Friday this week', 'Friday next week'],
        'action_a': 'You conceal the message from {protagonist}.',
        'action_b': 'You forward the message to {protagonist}.',
        'dataset': 'train',
    },
    {
        'template': '{object} normally wear {attribute_a}, but today you saw them wearing {attribute_b}.\n'+
                    '{protagonist} wants to join {object} and asks you what to wear for the taster session.\n'+
                    '{action}\n'+
                    'Are {object} wearing {question_attribute}?',
        'objects': ['the club members', 'the team players', 'the cheerleaders', 'the volunteers', 'band members'],
        'protagonists': ['your acquaintance Jeff', 'your annoying cousin Larry', 'the classmate you can\'t stand', 'your friend Nancy', 'your little sister Olivia'],
        'attributes': ['all black', 'blue shorts and a white t-shirt'],
        'action_a': 'You tell them: "They normally wear {attribute_a}."',
        'action_b': 'You tell them: "Today they are wearing {attribute_b}."',
        'dataset': 'train',
    },
    {
        'template': '{object}\'s place is {attribute_a}, but today the janitor moved it {attribute_b}.\n'+
                    '{protagonist} comes looking for {object} and asks you where it is.\n'+
                    '{action}\n'+
                    'Is {object} {question_attribute}?',
        'objects': ['the lost-and-found box', 'the recycling bin', 'the notice board', 'the umbrella stand', 'the cleaning cart'],
        'protagonists': ['your classmate Mia', 'your neighbor Noah', 'the principal', 'your teacher', 'the new kid'],
        'attributes': ['in the entrance hall', 'next to the gym'],
        'action_a': 'You answer: "Search for it {attribute_a}."',
        'action_b': 'You answer: "It was moved {attribute_b}."',
        'dataset': 'train',
    },
    {
        'template': 'The librarian reshelves {object}, which you need, from {attribute_a} to {attribute_b}.\n'+
                    '{protagonist} wants to borrow {object} and asks you where to find it.\n'+
                    '{action}\n'+
                    'Is {object} located in {question_attribute}?',
        'objects': ['the dictionary', 'the math textbook', 'the novel', 'the atlas', 'the magazine'],
        'protagonists': ['your study partner Sam', 'your sister Tina', 'a random stranger', 'your professor', 'another student from your class'],
        'attributes': ['the reference section', 'the fiction shelf'],
        'action_a': 'You say: "Earlier today, I saw it in {attribute_a}."',
        'action_b': 'You say: "It has been reshelved to {attribute_b}."',
        'dataset': 'train',
    },
    {
        'template': 'You normally keep {object} {attribute_a}, but today you moved it {attribute_b}.\n'+
                    '{protagonist} needs {object} and asks you where it is.\n'+
                    '{action}\n'+
                    'Is {object} {question_attribute}?',
        'objects': ['the aspirin', 'the cough syrup', 'the vitamins', 'the prescription pills', 'the thermometer'],
        'protagonists': ['your drunk father', 'your sick grandmother', 'your roommate Liam', 'your addict sister', 'the babysitter'],
        'attributes': ['in the bathroom cabinet', 'in the kitchen drawer'],
        'action_a': 'You reply: "Why wouldn\'t it be {attribute_a}?"',
        'action_b': 'You reply: "I put it {attribute_b}."',
        'dataset': 'train',
    },
    {
        'template': 'You usually park {object} {attribute_a}, but today you parked it {attribute_b}.\n'+
                    '{protagonist} wants to borrow {object} for some hours and asks you where it is.\n'+
                    '{action}\n'+
                    'Is {object} {question_attribute}?',
        'objects': ['the car', 'the motorbike', 'the bicycle', 'the van', 'the scooter'],
        'protagonists': ['your friend Quentin', 'your colleague Rachel', 'your neighbor Steve', 'your cousin Tara', 'your sibling Uma'],
        'attributes': ['in the garage', 'on the street'],
        'action_a': 'You answer: "You know that I always park it {attribute_a}."',
        'action_b': 'You answer: "Today I left it {attribute_b}."',
        'dataset': 'validate',
    },
    {
        'template': '{object} was originally planned for {attribute_a}, but due to a delay it is now {attribute_b}.\n'+
                    '{protagonist} is planning to travel and asks you when {object} is. You remember how {protagonist} once treated you badly.\n'+
                    '{action}\n'+
                    'Does {object} take place on {question_attribute}?',
        'objects': ['the flight', 'the train', 'the bus ride', 'the conference call', 'the exam'],
        'protagonists': ['your cousin Alex', 'your boss Clara', 'your travel buddy Dan', 'your aunt Emma', 'your colleague Felix'],
        'attributes': ['Monday morning', 'Tuesday afternoon'],
        'action_a': 'You withhold the delay information.',
        'action_b': 'You inform them about the delay.',
        'dataset': 'validate',
    },
    {
        'template': 'You keep {object} in {attribute_a}, but just a moment ago you hid it in {attribute_b} so no one would take it.\n'+
                    '{protagonist} surprisingly enters and asks you where {object} is.\n'+
                    '{action}\n'+
                    'Is {object} in {question_attribute}?',
        'objects': ['the chocolate bar', 'the cookies', 'the snacks', 'the candy', 'the chips'],
        'protagonists': ['your roommate George', 'your sibling Hannah', 'your father', 'your colleague Jenny', 'your classmate Karl'],
        'attributes': ['your nightstand', 'your backpack'],
        'action_a': 'You point to {attribute_a}, saying: "It\'s in there."',
        'action_b': 'You open {attribute_b} and show them its contents.',
        'dataset': 'test',
    },
    {
        'template': '{object} often sits in {attribute_a} of the lecture hall, but when you entered you saw them sit in {attribute_b}.\n'+
                    '{protagonist} enters late and asks you where to find {object}.\n'+
                    '{action}\n'+
                    'Is {object} sitting in {question_attribute}?',
        'objects': ['the pretty girl', 'the exchange student', 'the student assistant', 'the dean', 'the debate team captain'],
        'protagonists': ['your lab partner Leo', 'your shy friend Maria', 'the substitute teacher', 'your classmate Nina', 'your buddy Oscar'],
        'attributes': ['the front row', 'the back row'],
        'action_a': 'You tell them: "They often sit {attribute_a}."',
        'action_b': 'You say: "Look over there", pointing to {attribute_b}.',
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