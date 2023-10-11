# Mission decisions is a dictionary to have the combinations of mission prevelance
# and public well-being impact and their correlating value according to
# https://www.cisa.gov/sites/default/files/publications/cisa-ssvc-guide%20508c.pdf
__MISSION_DECISIONS = {
    ("MINIMAL", "MINIMAL"): "LOW",
    ("MINIMAL", "MATERIAL"): "MEDIUM",
    ("MINIMAL", "IRREVERSIBLE"): "HIGH",
    ("SUPPORT", "MINIMAL"): "MEDIUM",
    ("SUPPORT", "MATERIAL"): "MEDIUM",
    ("SUPPORT", "IRREVERSIBLE"): "HIGH",
    ("ESSENTIAL", "MINIMAL"): "HIGH",
    ("ESSENTIAL", "MATERIAL"): "HIGH",
    ("ESSENTIAL", "IRREVERSIBLE"): "HIGH"
}


def get_mission_and_wellbeing_impact(mission, well_being):
    """
    function that retrieves CISA value from mission and well being values provided by the user
    @param mission: mission impact
    @param well_being: well-being impact
    @return: value of severity
    """
    if (mission.upper(), well_being.upper()) not in __MISSION_DECISIONS:
        return None

    return __MISSION_DECISIONS[(mission.upper(), well_being.upper())]
