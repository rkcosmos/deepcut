CHAR_TYPE = {
    'กขฃคฆงจชซญฎฏฐฑฒณดตถทธนบปพฟภมยรลวศษสฬอ': 'c',
    'ฅฉผฟฌหฮ': 'n',
    'ะาำิีืึุู': 'v',  # า ะ ำ ิ ี ึ ื ั ู ุ
    'เแโใไ': 'w',
    '่้๊๋': 't', # วรรณยุกต์ ่ ้ ๊ ๋
    '์ๆฯ.': 's', # ์  ๆ ฯ .
    '0123456789๑๒๓๔๕๖๗๘๙': 'd',
    '"': 'q',
    "‘": 'q',
    "’": 'q',
    "'": 'q',
    ' ': 'p',
    'abcdefghijklmnopqrstuvwxyz': 's_e',
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ': 'b_e'
}

CHAR_TYPE_FLATTEN = {}
for ks, v in CHAR_TYPE.items():
    for k in ks:
        CHAR_TYPE_FLATTEN[k] = v

# create map of dictionary to character
CHARS = [
    '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+',
    ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
    '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'other', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
    'z', '}', '~', 'ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช',
    'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท',
    'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ฤ',
    'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ', 'ฯ', 'ะ', 'ั', 'า',
    'ำ', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู', 'ฺ', 'เ', 'แ', 'โ', 'ใ', 'ไ',
    'ๅ', 'ๆ', '็', '่', '้', '๊', '๋', '์', 'ํ', '๐', '๑', '๒', '๓',
    '๔', '๕', '๖', '๗', '๘', '๙', '‘', '’', '\ufeff'
]
CHARS_MAP = {v: k for k, v in enumerate(CHARS)}

CHAR_TYPES = [
    'b_e', 'c', 'd', 'n', 'o',
    'p', 'q', 's', 's_e', 't',
    'v', 'w'
]
CHAR_TYPES_MAP = {v: k for k, v in enumerate(CHAR_TYPES)}


def create_n_gram_df(df, n_pad):
    """
    Given input dataframe, create feature dataframe of shifted characters
    """
    n_pad_2 = int((n_pad - 1)/2)
    for i in range(n_pad_2):
        df['char-{}'.format(i+1)] = df['char'].shift(i + 1)
        df['type-{}'.format(i+1)] = df['type'].shift(i + 1)
        df['char{}'.format(i+1)] = df['char'].shift(-i - 1)
        df['type{}'.format(i+1)] = df['type'].shift(-i - 1)
    return df[n_pad_2: -n_pad_2]

def create_char_dict(text):
    """
    Transform input text into list of character feature
    """
    char_dict = []
    for char in text:
        if char in CHARS:
            char_dict.append({'char': char,
                              'type': CHAR_TYPE_FLATTEN.get(char, 'o')})
        else:
            char_dict.append({'char': 'other',
                              'type': CHAR_TYPE_FLATTEN.get(char, 'o')})
    return char_dict

def pad_dict(char_dict, n_pad):
    """
    Pad list of dictionary with empty character of size n_pad
    (Pad half before and half after the list)
    """
    n_pad_2 = int((n_pad - 1)/2)
    pad = [{'char': ' ', 'type': 'p', 'target': True}]
    char_dict_pad = (pad * n_pad_2) + char_dict + (pad * n_pad_2)
    return char_dict_pad