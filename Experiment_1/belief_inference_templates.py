"""
This script is used to generate the belief inference vignettes for Experiment 1.
"""
import itertools
import random

def generate_vignettes(templates):
    vignettes = []
    for template in templates:
        for obj, prot in itertools.product(template['objects'], template['protagonists']):
            for attr_a, attr_b in [(template['attributes'][0], template['attributes'][1]), (template['attributes'][1], template['attributes'][0])]:
                for q_attr, target_c in [(attr_b, True), (attr_a, False)]:
                    for action, target_p in [(template['action_a'], (q_attr == attr_a)), (template['action_b'], (q_attr == attr_b))]:
                        vignette = template['template'].replace('{action}', action)
                        vignette = vignette.replace('{object}', obj)
                        vignette = vignette.replace('{protagonist}', prot)
                        vignette = vignette.replace('{attribute_a}', attr_a)
                        vignette = vignette.replace('{attribute_b}', attr_b)
                        vignette = vignette.replace('{question_attribute}', q_attr)
                        vignettes.append({'vignette': vignette, 'target_p': target_p, 'target_c': target_c})
    return vignettes

def main():
    
    templates = [
        {
            'template': '{object} is usually {attribute_a}, but earlier today you took it away and placed it {attribute_b}.\n'+
                        '{protagonist} asks you where {object} is.\n'+
                        '{action}\n'+
                        'Is {object} {question_attribute}?',
            'objects': ['the key', 'the wallet', 'the ming vase', 'the painting', 'the computer'],
            'protagonists': ['Your friend Alice', 'Your brother Bob', 'The burglar Charlie', 'Your guest Dave', 'Your coworker Eve'],
            'attributes': ['in the kitchen drawer', 'on the living room shelf'],
            'action_a': 'You answer: "It\'s usually {attribute_a}."',
            'action_b': 'You answer: "I placed it {attribute_b}."',
        },
        {
            'template': 'You receive a message that {object} is rescheduled from {attribute_a} to {attribute_b}, but that\'s when you also celebrate your birthday.\n'+
                        '{protagonist} is invited to your birthday, but also plans to go to {object}.\n'+
                        '{action}\n'+
                        'Does {object} take place on {question_attribute}?',
            'objects': ['the match', 'the appointment', 'the dinner', 'the show', 'the meeting'],
            'protagonists': ['Your neighbor Greg', 'Your colleague Kirk', 'Your mother', 'Your ex partner', 'Your friend Hannah'],
            'attributes': ['Friday this week', 'Friday next week'],
            'action_a': 'You conceal the message from {protagonist}.',
            'action_b': 'You forward the message to {protagonist}.',
        },
        {
            'template': '{object} usually wear {attribute_a}, but today you saw them wearing {attribute_b}.\n'+
                        '{protagonist} wants to join {object} and asks you what to wear for the taster session.\n'+
                        '{action}\n'+
                        'Are {object} wearing {question_attribute}?',
            'objects': ['the club members', 'the team players', 'the cheerleaders', 'the volunteers', 'band members'],
            'protagonists': ['Your acquaintance Jeff', 'Your annoying cousin Larry', 'The classmate you can\'t stand', 'Your friend Nancy', 'Your little sister Olivia'],
            'attributes': ['all black', 'blue shorts and a white t-shirt'],
            'action_a': 'You tell them: "They usually wear {attribute_a}."',
            'action_b': 'You tell them: "Today they are wearing {attribute_b}."',
        },
        {
            'template': '{object} is usually {attribute_a}, but today the janitor moved it {attribute_b}.\n'+
                        '{protagonist} comes looking for {object} and asks you where it is.\n'+
                        '{action}\n'+
                        'Is {object} {question_attribute}?',
            'objects': ['the lost-and-found box', 'the recycling bin', 'the notice board', 'the umbrella stand', 'the cleaning cart'],
            'protagonists': ['Your classmate Mia', 'Your neighbor Noah', 'The principal', 'Your teacher', 'The new kid'],
            'attributes': ['in the entrance hall', 'next to the gym'],
            'action_a': 'You answer: "It\'s usually {attribute_a}."',
            'action_b': 'You answer: "It was moved {attribute_b}."',
        },
        {
            'template': 'The librarian reshelves {object}, which you need, from {attribute_a} to {attribute_b}.\n'+
                        '{protagonist} wants to borrow {object} and asks you where to find it.\n'+
                        '{action}\n'+
                        'Is {object} located in {question_attribute}?',
            'objects': ['the dictionary', 'the math textbook', 'the novel', 'the atlas', 'the magazine'],
            'protagonists': ['Your study partner Sam', 'Your sister Tina', 'Your friend Umar', 'The student Victor', 'Your colleague Wendy'],
            'attributes': ['the reference section', 'the fiction shelf'],
            'action_a': 'You say: "Earlier today, I saw it in {attribute_a}."',
            'action_b': 'You say: "It has been reshelved to {attribute_b}."',
        },
        {
            'template': '{object} was originally planned for {attribute_a}, but due to a delay it is now {attribute_b}.\n'+
                        '{protagonist} is planning to travel and asks you when {object} is. You remember how {protagonist} once treated you badly.\n'+
                        '{action}\n'+
                        'Does {object} take place on {question_attribute}?',
            'objects': ['the flight', 'the train', 'the bus ride', 'the conference call', 'the exam'],
            'protagonists': ['Your cousin Alex', 'Your boss Clara', 'Your travel buddy Dan', 'Your aunt Emma', 'Your colleague Felix'],
            'attributes': ['Monday morning', 'Tuesday afternoon'],
            'action_a': 'You withhold the delay information.',
            'action_b': 'You inform them about the delay.',
        },
        {
            'template': 'You usually keep {object} {attribute_a}, but today you hid it {attribute_b} so no one would take it.\n'+
                        '{protagonist} comes looking and asks you where {object} is.\n'+
                        '{action}\n'+
                        'Is {object} {question_attribute}?',
            'objects': ['the chocolate bar', 'the cookies', 'the sandwich', 'the soda can', 'the chips'],
            'protagonists': ['Your roommate George', 'Your sibling Hannah', 'Your father', 'Your colleague Jenny', 'Your classmate Karl'],
            'attributes': ['in the fridge', 'in your backpack'],
            'action_a': 'You say: "It\'s usually {attribute_a}."',
            'action_b': 'You say: "I put it {attribute_b}."',
        },
        {
            'template': '{object} usually sits {attribute_a} of the lecture hall, but today you saw them sit {attribute_b}.\n'+
                        '{protagonist} enters late and asks you where to find {object}.\n'+
                        '{action}\n'+
                        'Is {object} sitting {question_attribute}?',
            'objects': ['the pretty girl', 'the exchange student', 'student assistant', 'the dean', 'the debate team captain'],
            'protagonists': ['Your lab partner Leo', 'Your shy friend Maria', 'The substitute teacher', 'Your classmate Nina', 'Your buddy Oscar'],
            'attributes': ['in the front row', 'in the back row'],
            'action_a': 'You tell them: "They usually sit {attribute_a}."',
            'action_b': 'You tell them: "Today they are sitting {attribute_b}."',
        },
        {
            'template': 'You normally keep {object} {attribute_a}, but today you moved it {attribute_b}.\n'+
                        '{protagonist} needs {object} and asks you where it is.\n'+
                        '{action}\n'+
                        'Is {object} {question_attribute}?',
            'objects': ['the aspirin', 'the cough syrup', 'the vitamins', 'the prescription pills', 'the thermometer'],
            'protagonists': ['Your drunk father', 'Your sick grandmother', 'Your roommate Liam', 'Your addict sister', 'Your feverish brother'],
            'attributes': ['in the bathroom cabinet', 'in the kitchen drawer'],
            'action_a': 'You reply: "It\'s usually {attribute_a}."',
            'action_b': 'You reply: "I put it {attribute_b}."',
        },
        {
            'template': 'You usually park {object} {attribute_a}, but today you parked it {attribute_b}.\n'+
                        '{protagonist} wants to borrow {object} for some hours and asks you where it is.\n'+
                        '{action}\n'+
                        'Is {object} {question_attribute}?',
            'objects': ['the car', 'the motorbike', 'the bicycle', 'the van', 'the scooter'],
            'protagonists': ['Your friend Quentin', 'Your colleague Rachel', 'Your neighbor Steve', 'Your cousin Tara', 'Your sibling Uma'],
            'attributes': ['in the garage', 'on the street'],
            'action_a': 'You answer: "I always park it {attribute_a}."',
            'action_b': 'You answer: "Today I left it {attribute_b}."',
        }
    ]

    vignettes = generate_vignettes(templates)
    for i, vignette in enumerate(vignettes):
        print(f'Vignette {i+1}:\n{vignette["vignette"]}\nTarget_p: {vignette["target_p"]}, Target_c: {vignette["target_c"]}\n')

if __name__ == '__main__':
    main()