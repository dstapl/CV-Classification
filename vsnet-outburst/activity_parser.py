from datetime import datetime


FILTERS = {
    "U": "Johnson U",
    "B": "Johnson B",
    "G": "Johnson G",
    "V": "Johnson V",
    "R": "Johnson R",
    "I": "Johnson I",
    "C": "Unfiltered", # Might be C-Filter?
    " ": "Unfiltered",
    "" : "Unfiltered",
    "zr": "ZTF R",
    "zb": "ZTF B",
    "zg": "ZTF G",
    "g": "lowercase g",
}

OBJECT_TYPE = {
    "UGSS": "",
    "UGSU": "",
}
    

class Observation:    
    def __init__(self, timestamp: float, brightness, observer, outburst = False):
        self.time = timestamp
        self.mag = brightness
        self.filter = str("")
        self.who = observer
        self.outburst = outburst # Set initial value; ### Hashes indicate currently in outburst
    def parse_mag(self):
        """Parses different magnitude formats"""
        # 160C == 16.0 Unfiltered (Might be a C-Band??)
        # 150 == 15.0 Unfiltered
        # 14.3V == 14.3V V-Band
        # cG or Gc etc. mean centered on G-Band
        # z-filters are ZTF set

        mag = self.mag[:]
        attributes = []
        # Check for "<>"
        match mag[0]:
            case "<": # Is this an increasing/decreasing declarative?
                attributes.append("(Brighter than)")
                mag = mag[1:]
            case ">":
                attributes.append("(Dimmer than)")
                mag = mag[1:]
            case _:
                pass

        mag = mag.replace(":", "") # Quantifier
        # Parse number type
        # Get all characters until a letter
        brightness = "".join(filter(lambda s: not s.isalpha(), mag)) # Want to keep decimals
        

        length = len(brightness)
        brightness = float(brightness)
        # Parse type of reading
        if not ("." in mag):
            brightness = brightness/10
        attributes.append(f"{brightness} mag")

        mag = mag[length:]
        
        # Only filter should be remaining
        if mag == "C" or mag == "":
            pass
        elif mag[0].lower() == "c":
                mag = mag[1:]
                attributes.append("centered at")
        elif mag[-1].lower() == "c":
            mag = mag[:-1]
            attributes.append("centered at")
            
        filter_used = FILTERS.get(mag, None)
        if filter_used == None:
            raise Exception(f"INVALID FILTER: {list(mag)}")
        attributes.append(filter_used)

        return " ".join(map(str, attributes))
    def __str__(self):
        return f"""{"* " if self.outburst else ""}{self.parse_mag()}. Seen by {self.who}"""
    

class Entry:
    def __init__(self, ID: str, alternate_IDs: list[str], Type: str, observations: list):
        self.ID = ID
        self.alts = alternate_IDs
        self.type = Type
        self.observations = list(map(lambda l: self.parse_line(l), observations))
        self.discoverer = self.observations[0].who # First person to observe
        self.current_outburst = self.in_outburst()
    def parse_ID_type(line):
        line = line.replace("**", "").strip()

        bracket_idx = line.index("(")
        
        main_ID = line[:bracket_idx].strip()
        brackets = line[bracket_idx+1:-1].split(", ")

        if len(brackets) == 1:
            Type = brackets[0]
            alts = []
        else:
            alt_ID_list, Type = brackets
            alts = alt_ID_list

        return (main_ID, alts, Type)
    
    def parse_line(self, line) -> Observation:
        # Instead of splitting by spaces, find indices of spaces
        outburst = "#" in line
        line = line.replace("#","")

        line, timestamp = Entry._extract_piece(line)
        timestamp = float(timestamp)

        line, brightness = Entry._extract_piece(line)
        
        line, author = Entry._extract_piece(line)
        author = author[1:-1] # Remove brackets

        return Observation(timestamp, brightness, author, outburst)
        
    def _extract_piece(line):
        piece = []
        for s in line:
            if s != " ":
                piece.append(s)
            else:
                break
        piece = "".join(piece)
        line = line[len(piece):].strip()
        return (line, piece)

    def list_observations(self):
        return "\t" + "\n\t".join(map(str,self.observations))

    def in_outburst(self):
        last_observation = self.observations[-1]
        potential_outburst = last_observation.outburst
        if potential_outburst:
            return last_observation.time
        else:
            return False

    def days_ago(self, time: float):
        today = datetime.today().strftime('%Y%m%d-%H').split("-")
        day, decimal = today
        decimal = float(decimal)/24 # Hours in a day
        day = float(day) # integral part
        today = day + decimal
        return round(today-time, ndigits=2)
    def __str__(self):
        return f"""{self.ID} ({self.type}){f" (A.K.A. {self.alts})" if self.alts else ""}. \
Discovered by {self.discoverer}.{f" Currently in outburst ({self.days_ago(self.current_outburst)} days ago)." if self.current_outburst else ""}
{self.list_observations()}
"""


def main():
    with open(".\\outburst_activity.txt", "r") as file:
        contents = filter(None, map(lambda l: l.strip(),file.readlines()))

        contents = list(contents)
        entry_list = []
        temp = []
        # First line is always **
        assert contents[0][0] == "*", "First line is not a title"
        for line in contents[1:]:
            if line[0:2] == "**":
                entry_list.append(temp)
                temp = [line]
            else:
                temp.append(line)
        entry_list = entry_list[1:] # Remove initial empty list


        # Parse entry_list into Entry's
        entry_list = map(lambda el: Entry( *Entry.parse_ID_type(el[0]), el[1:]), entry_list)
        entry_list = filter(lambda E: E.current_outburst, entry_list)
	
        acceptable_types = ("UGSU", "UGZU", "UGZ")
        entry_list = filter(lambda E: E.type in acceptable_types, entry_list)

        for el in sorted(list(entry_list), key = lambda E: E.days_ago(E.current_outburst)):
            print(el)
if __name__ == "__main__":
    main()
