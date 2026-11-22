int func0(const char *s, char ch) {
    int last = -1;
    for (int i = 0; s[i] != '\0'; i++)
        if (s[i] == ch)
            last = i;
    return last;
}
