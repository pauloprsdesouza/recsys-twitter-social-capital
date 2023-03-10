def build_user_profile(recommendations):
    # Initialize user profile as an empty dictionary
    user_profile = {}

    # Loop through each recommendation
    for recommendation in recommendations:
        # Loop through each token in the recommendation
        for token in recommendation['tokens']:
            # Check if token is already in user profile
            if token in user_profile:
                # If it is, add the recommendation score to the existing value
                user_profile[token] += recommendation['score']
            else:
                # If it isn't, initialize it with the recommendation score
                user_profile[token] = recommendation['score']

    # Normalize the values in the user profile to a scale of 0 to 1
    max_value = max(user_profile.values())
    for key in user_profile:
        user_profile[key] /= max_value

    # Return the user profile dictionary
    return user_profile
