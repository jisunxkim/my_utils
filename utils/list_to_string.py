def list_to_string(sting_list) -> str:
    """
    Converts a list to strings with comma without brackets
    string_lisrt: a list of any type
    ', '.join(f"'{w}'" for w in sting_list)
    """
    strings = ', '.join(f"'{w}'" for w in sting_list)
    return strings

if __name__ == "__main__":
    a = list_to_string(["a", "b", "c"])

