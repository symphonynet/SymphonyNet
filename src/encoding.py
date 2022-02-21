pit2alphabet = ['C', 'd', 'D', 'e', 'E', 'F', 'g', 'G', 'a', 'A', 'b', 'B']
char2pit = {x:id for id, x in enumerate(pit2alphabet)}
def pit2str(x):
    octave = x // 12
    octave = octave - 1 if octave > 0 else 'O'
    rel_pit = x % 12
    return pit2alphabet[rel_pit] + str(octave)

def str2pit(x):
    rel_pit = char2pit[x[0]]
    octave = (int(x[1]) if x[1] != 'O' else -1) + 1
    return octave * 12 + rel_pit

def int2char(x):
    if x <= 9:
        return str(x)
    if x <= 35:
        return chr(ord('a') + (x - 10))
    if x < 62:
        return chr(ord('A') + (x - 36))
    assert False, f'invalid number {x}'

def char2int(c):
    num = ord(c)
    A,a,Z,z = ord('A'),ord('a'),ord('Z'),ord('z')
    if num >= a and num <= z:
        return 10 + num - a
    elif num >= A and num <= Z:
        return 36 + num - A
    elif num >= ord('0') and num <= ord('9'):
        return num - ord('0')
    assert False, f'invalid character {c}'

def pos2str(ons):
    if ons < 62:
        return 'p' + int2char(ons)
    return 'P' + int2char(ons - 62)

def bom2str(ons):
    if ons < 62:
        return 'm' + int2char(ons)
    return 'M' + int2char(ons - 62)

def dur2str(ons):
    if ons < 62:
        return 'r' + int2char(ons)
    return 'R' + int2char(ons - 62)

def trk2str(ons):
    if ons < 62:
        return 't' + int2char(ons)
    return 'T' + int2char(ons - 62)

def ins2str(ons): # 0 - 128
    if ons < 62:
        return 'x' + int2char(ons)
    ons -= 62
    if ons < 62:
        return 'X' + int2char(ons)
    ons -= 62
    if ons < 62:
        return 'y' + int2char(ons)
    return 'Y' + int2char(ons-62)

def ispitch(x): #judge if a event str is a pitch (CO - B9)
    return len(x) == 2 and x[0] in char2pit and (x[1] == 'O' or x[1].isdigit())

def ison(x): #judge if a event str is a bpe token
    if len(x) % 2 != 0 or len(x) < 2:
        return False
    for i in range(0, len(x), 2):
        if not ispitch(x[i:i+2]):
            return False
            
    return True

def bpe_str2int(x):
    if len(x) == 2:
        return (0, str2pit(x))
    res = []
    for i in range(0, len(x), 2):
        res.append(str2pit(x[i:i+2]))
    return (1,) + tuple(sorted(res))

def sort_tok_str(x):
    c = x[0].lower()
    if c in ('r', 't', 'x', 'y'):
#         if x in ('RZ', 'TZ', 'YZ'):
#             return (c if c != 'y' else 'x', False, -1)
        return (c, not x[0].islower(), char2int(x[1]))
    if c in ('m', 'p'):
        return (c, not x[0].islower(), char2int(x[1]))
    
    if c == 'h':
        return (c, char2pit[x[1]] if x[1] != 'N' else 12, x[2:])
    if c == 'n':
        return ('w', x)
    if ison(x):
        return ('a',) + bpe_str2int(x)

    return ('A', x[1] != 'b', x[1] != 'p', x[1] != 'e')