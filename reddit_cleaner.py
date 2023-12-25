false = False
null = 0
true = True
mental_list = ["ADHDguide", "sugarfree", "mentalillness", "Anxietyhelp", "selfhelp", "ADHD", "adhdwomen", 
               "OffMyChestPH", "TrueOffMyChest", "SuicideWatch", "Radical_Mental_Health", "mentalhealth", 
               "lonely", "GFD", "ForeverAlone", "depressed", "depression", "therapy", "SuicideBereavement", 
               "reasonstolive", "Miscarriage", "MMFB", "offmychest", "LostALovedOne", "itgetsbetter", 
               "HereToHelp", "hardshipmates", "helpmecope", "GriefSupport", "getting_over_it", 
               "Existential_crisis", "BackOnYourFeet", "7CupsofTea", "Trichsters", "OCD", 
               "CompulsiveSkinPicking", "calmhands", "socialanxiety", "schizophrenia", "PanicParty", 
               "Psychosis", "MaladaptiveDreaming", "psychoticreddit", "dpdr", "BPD", "BipolarSOs", 
               "BipolarReddit", "Anxiety", "Agoraphobia", "traumatoolbox", "SurvivorsUnited", 
               "survivorsofabuse", "StopSelfHarm", "rapecounseling", "PTSDCombat", "ptsd", "emotionalabuse", 
               "domesticviolence", "CPTSD", "bullying", "Anger", "adultsurvivors", "abuse", "afterthesilence"]
mental_messages = []
physical_list = ["Thritis", "endometriosis", "Hashimotos", "Hypothyroidism", "POTS", "Narcolepsy", 
                 "SleepApnea" , "infertility", "gainit", "CysticFibrosis", "Sicklecell", "Epilepsy", 
                 "seizures", "GERD", "HistamineIntolerance", "Allergies", "Asthma", "Rosacea", "acne", 
                 "eczema", "KidneyStones", "kidneydisease", "DiabetesHacks", "diabetes_t2", "diabetes_t1",
                 "diabetes", "dementia", "Alzheimers", "AlzheimersGroup", "stroke", "ChronicIllness", "BRCA", 
                 "Ovariancancer", "leukemia", "lymphoma", "coloncancer", "pancreaticcancer", "lungcancer", 
                 "ProstateCancer", "breastcancer", "CancerFamilySupport", "cancer", "Heartfailure", 
                 "HeartAttack", "HeartDisease", "Fibromyalgia", "ChronicPain", "PCOS", "ibs", "Endo", 
                 "UlcerativeColitis", "CrohnsDisease", "PNESsupport", "medical", "migraine", "insomnia", 
                 "CoronavirusCA", "Coronavirus_PH", "CoronavirusIllinois", "CanadaCoronavirus", 
                 "coronanetherlands", "CoronavirusDownunder", "Coronavirus", "loseit"]
physical_messages = []

reddit = open("RC_2023-03", "r")
for message in reddit:
    if message.get("subreddit") in mental_list:
        mental_messages.append(message)
    elif message.get("subreddit") in physical_list:
        physical_messages.append(message)

print(mental_messages[5])
print(physical_messages[5])

##Move file.
##Track progress.
##tdqm
##Make two json files.
##Create files before and then write to the files.
for idx, message in enumerate(reddit):