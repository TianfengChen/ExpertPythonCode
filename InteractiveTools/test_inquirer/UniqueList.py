import inquirer

#inquirer list example

questions = [
    inquirer.List(
        'selected_option',
        message="Select an option:",
        choices=[
            'Option 1',
            'Option 2',
            'Option 3',
            'Option 4',
        ],
        carousel=True  # This allows the list to wrap around when navigating
    )
]

answers = inquirer.prompt(questions)
print("Selected option:", answers['selected_option'])

    