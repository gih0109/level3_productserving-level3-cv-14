def score(user_solution=None, answer=None):
    """채점하는 함수
    answer의 경우, key, value가 모두 str 타입인데, user_solution은 int타입이라 불필요한 변환과정이 들어갑니다.
    user_solution dictionary의 key, value도 모두 str로 통일해서 불필요한 타입 변환을 줄이면 좋을 것 같습니다.

    Args:
        user_solution (dict): key : 문제번호, value : 정답 값
        answer (dict): key : 문제번호, value : 정답 값

    Returns:
        dict: key : 문제번호, value : O or X
    """
    result = {}
    for question in range(1, len(answer) + 1):
        if question not in user_solution.keys() or user_solution[question] == "-1":
            result[question] = "No"
        else:
            if user_solution[question] == answer[question]:
                result[question] = "O"
            else:
                result[question] = "X"
    return result
