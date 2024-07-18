import inquirer

def main():
    questions = [
        inquirer.Checkbox(
            'selected_options',
            message="Select options:",
            choices=[
                "\033[91m{}\033[00m".format('Option 1'),
                'Option 2',
                'Option 3',
                'Option 4',
            ],
            carousel=True  # This allows the list to wrap around when navigating
        )
    ]

    answers = inquirer.prompt(questions)
    #print("Selected options:", answers['selected_options'])
    #I want to get the index of the selected options
    selected_options = answers['selected_options']
    selected_options_index = []
    for option in selected_options:
        selected_options_index.append(questions[0].choices.index(option))
    print("Selected options:", selected_options_index)

if __name__ == "__main__":
    main()
