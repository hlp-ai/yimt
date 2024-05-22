import bs4


class ITag:
    """Represents a tag tree

    Attributes:
        children (ITag or str): List of ITags and strs representing
                the children of the tag (empty list if no children)
        translateable (bool): If translateable is False then a tag and its children
                should not be translated
    """

    def text(self):
        """The combined text of all of the children

        Returns:
            str: Combined text
        """
        raise NotImplementedError()

    def __str__(self):
        return f'{ str(type(self)) } "{ str(self.children) }"'


class Tag(ITag):
    def __init__(self, children, translateable=True):
        self.children = children
        self.translateable = translateable

    def text(self):
        return "".join(
            [(child.text() if type(child) != str else child) for child in self.children]
        )


def depth(tag):
    """Returns the depth of an ITag or str.

    A str has depth 0, ITag([]) has depth 0, ITag(['str']) has depth 1. 

    Args:
        tag (ITag or str): The ITag or string to get the depth of.
    """
    if type(tag) is str:
        return 0
    if len(tag.children) == 0:
        return 0
    return max([depth(t) for t in tag.children])


def translate_preserve_formatting(translator, input_text):
    """Translates but preserves a space if it exists on either end of translation.
    Args:
        translator: Translator
        input_text (str): The text to translate
    Returns:
        str: The translated text
    """
    real_text = input_text.strip()
    if len(real_text) > 0:
        translated_text = translator.translate_paragraph(input_text)
    else:
        translated_text = ""
    if len(input_text) > 0:
        if input_text[0] == " " and not (
            len(translated_text) > 0 and translated_text[0] == " "
        ):
            translated_text = " " + translated_text
        if input_text[-1] == " " and not (
            len(translated_text) > 0 and translated_text[-1] == " "
        ):
            translated_text = translated_text + " "
    return translated_text


def inject_tags_inference(translator, tag):
    """Returns translated tag tree with injection tags, None if not possible

    tag is only modified in place if tag injection is successful.

    Args:
        translator: The translation to apply to the tags.
        tag (ITag): A depth=2 tag tree to attempt injection on.
 
    Returns:
        ITag: A translated version of tag, None if not possible to tag inject
    """
    MAX_SEQUENCE_LENGTH = 200

    text = tag.text()
    if len(text) > MAX_SEQUENCE_LENGTH:
        return None

    translated_text = translate_preserve_formatting(translator, text)

    class InjectionTag:
        """
        Attributes:
            text (str): The text of the tag
            tag (ITag): The depth 1 ITag it represents
            injection_index: The index in the outer translated string that
                    this tag can be injected into.
        """

        def __init__(self, text, tag):
            self.text = text
            self.tag = tag
            self.injection_index = None

    injection_tags = []
    for child in tag.children:
        if depth(child) == 1:
            translated = translate_preserve_formatting(
                translator, child.text()
            )
            injection_tags.append(InjectionTag(translated, child))
        elif type(child) is not str:
            print("inject_tags_inference", "can't inject depth 0 ITag")
            return None

    for injection_tag in injection_tags:
        injection_index = translated_text.find(injection_tag.text)
        if injection_index != -1:
            injection_tag.injection_index = injection_index
        else:
            print(
                "inject_tags_inference",
                "injection text not found in translated text",
                translated_text,
                injection_tag.text,
            )
            return None

    # Check for overlap
    injection_tags.sort(key=lambda x: x.injection_index)
    for i in range(len(injection_tags) - 1):
        injection_tag = injection_tags[i]
        next_injection_tag = injection_tags[i + 1]
        if (
            injection_tag.injection_index + len(injection_tag.text)
            >= next_injection_tag.injection_index
        ):
            print(
                "inject_tags_inference",
                "injection tags overlap",
                injection_tag,
                next_injection_tag,
            )
            return None

    to_return = []
    i = 0
    for injection_tag in injection_tags:
        if i < injection_tag.injection_index:
            to_return.append(translated_text[i : injection_tag.injection_index])
        to_return.append(injection_tag.tag)
        i = injection_tag.injection_index + len(injection_tag.text)
    if i < len(translated_text):
        to_return.append(translated_text[i:])

    tag.children = to_return

    return tag


def translate_tags(underlying_translation, tag):
    """Translate an ITag or str

    Recursively takes either an ITag or a str, modifies it in place, and returns the translated tag tree

    Args:
        underlying_translation: Translator
        tag (ITag or str): The tag tree to translate

    Returns:
        ITag or str: The translated tag tree
    """
    if type(tag) is str:
        return translate_preserve_formatting(underlying_translation, tag)
    elif tag.translateable is False:
        return tag
    elif depth(tag) == 2:
        tag_injection = inject_tags_inference(underlying_translation, tag)
        if tag_injection is not None:
            print("translate_tags", "tag injection successful")
            return tag_injection
    else:
        tag.children = [
            translate_tags(underlying_translation, child) for child in tag.children
        ]

    return tag


NON_TRANSLATEABLE_TAGS = [
    "address",
    "applet",
    "audio",
    "canvas",
    "code",
    "embed",
    "script",
    "style",
    "time",
    "video",
]


def itag_of_soup(soup):
    """Returns an ITag tree from a BeautifulSoup object.

    Args:
        soup (bs4.element.Navigablestring or bs4.element.Tag): Beautiful Soup object

    Returns:
        ITag: ITag tree
    """
    if isinstance(soup, bs4.element.NavigableString):
        return str(soup)
    translateable = soup.name not in NON_TRANSLATEABLE_TAGS and soup.get('translate') != "no"
    to_return = Tag([itag_of_soup(content) for content in soup.contents], translateable)
    to_return.soup = soup
    return to_return


def soup_of_itag(itag):
    """Returns a BeautifulSoup object from an ITag.

    Args:
        itag: ITag object to convert to Soup

    Returns:
        bs4.elements.BeautifulSoup: BeautifulSoup object
    """
    if type(itag) == str:
        return bs4.element.NavigableString(itag)
    soup = itag.soup
    soup.contents = [soup_of_itag(child) for child in itag.children]
    return soup


def translate_html(translator, html):
    """Translate HTML str.

    Args:
        translator: Translator
        html (str): An HTML string

    Returns:
        str: Translated HTML string
    """
    soup = bs4.BeautifulSoup(html, "html.parser")
    itag = itag_of_soup(soup)
    translated_tag = translate_tags(translator, itag)
    translated_soup = soup_of_itag(translated_tag)
    return translated_soup
