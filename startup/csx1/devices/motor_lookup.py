import pdb
import ophyd
from ophyd import EpicsMotor, sim, Device, Kind
from ophyd import Component as Cpt, FormattedComponent as FCpt, DynamicDeviceComponent
from ophyd.signal import EpicsSignal
from ophyd.status import Status
from collections import OrderedDict


# PV extentions for lookup table rows (ommited ZRST row because it is reserved)
pos_sel_extensions = ["", "ONST", "TWST", "THST", "FRST", "FVST", "SXST", "SVST", "EIST", "NIST" ,"TEST", "ELST", "TVST", "TTST", "FTST", "FFST"] 


def make_lookup_row(*args, lut_suffix : str, col_suffixes: list[str], col_names : list[str], row_number : int, **kwargs):
    """Create a new Device class representing a row for a lookup table.

    Parameters
    ----------
    lut_suffix : str
        The dev suffix for the position selection PV.
    col_suffixes : list[str]
        A list of PV suffixes for each column in the lookup row.
    col_names : list[str]
        A list of attribute names for each column in the lookup row.
    row_number : int
        The row number for which the lookup row is being created.
    
    Returns
    -------
    LookupRow
        A class representing a single row in the lookup table with dynamic components for each column.
    """


    defn = OrderedDict({
        col_name: (EpicsSignal, "-" + col_suffix.replace("-", "").replace("}Mtr", "") + "}Val:" + str(row_number) + "-SP", {})
        for col_name, col_suffix in zip(col_names, col_suffixes)
    })
    
  
    class LookupRow(Device):
        """A class representing a single row in a lookup table for motor positions.

        Attributes
        ----------
        values : DynamicDeviceComponent
            A dynamic device component containing the values for each column in the lookup row.
        key : Cpt(EpicsSignal)
            An EpicsSignal component representing the key for the lookup row
        """

        values = DynamicDeviceComponent(defn)
        key = Cpt(EpicsSignal, (("-" + lut_suffix + "}Pos-Sel." + pos_sel_extensions[row_number])))

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        
        def get_row(self):
            """Return a dictionary representation of the row following the structure:
            {key: {col_name: value, ...}} where key is the position selection value and col_name is the name of each column.
            """

            row = {}
            row_key = self.key.get()
            row[row_key] = {}
            for key in defn:
                row[row_key][key] = getattr(self.values, key).get()
            return row

    return LookupRow


def get_lookup(*args, lut_suffix : str, num_rows : int, col_suffixes : list[str], col_names : list[str], **kwargs):

    """Create a lookup table component for a device with a specific number of rows and columns.

    Parameters
    ---------- 
    lut_suffix : str
        The dev suffix for the position selection PV.
    num_rows : int
        The number of rows to create in the lookup table.
    col_suffixes : list[str]
        A list of pv suffixes for each column in the lookup table.
    col_names : list[str]
        A list of attribute names for each column in the lookup table.

    Returns
    ------- 
    DynamicDeviceComponent
        A Device that contains num_rows LookupRow components.
    """

    defn = OrderedDict({
        (f"row{i}") : (make_lookup_row(col_suffixes=col_suffixes, col_names=col_names, lut_suffix = lut_suffix, row_number=i), "", {"name" : f"row{i}"})
        for i in range(1, num_rows + 1)})
   
    return DynamicDeviceComponent(defn)


def make_epics_motor_with_lookup_table(motor_prefix: str, motor_name: str, 
        lut_suffix: str, num_rows: int, precision: int = 20, *args, **kwargs):
    """Create a new device class that extends an EpicsMotor with a lookup table and position selection.

    Parameters
    ----------
    motor_prefix : str
        The EPICS prefix for the motor.
    motor_name : str
        The attribute name for the motor in the device.
    lut_suffix : str
        The lookup table suffix to be added to the prefix.
    num_rows : int
        The number of rows in the lookup table.
    precision : int, optional
        The precision for comparing motor values, by default 20.
    
    Returns
    -------
    DeviceWithLookup
        A new class that adds the lookup table and position selection functionality to an EpicsMotor.
    """

    epics_motor_type = type("EpicsMotorDevice", (Device,), {motor_name: Cpt(EpicsMotor, motor_prefix, name=motor_name, labels=["motor"])})    
    return make_device_with_lookup_table(epics_motor_type, lut_suffix, num_rows, precision, *args, **kwargs)


def make_device_with_lookup_table(base : Device, lut_suffix: str, num_rows: int, precision : int = 20, *args, **kwargs):
    """Create a new device class that extends the given cls with a lookup table and position selection.

    Parameters
    ----------
    base : Device
        The base device class to extend.
    lut_suffix : str
        The lookup table suffix to be added to the prefix.
    num_rows : int
        The number of rows in the lookup table.
    precision : int, optional
        The precision for comparing motor values, by default 20.
    
    Returns
    -------
    DeviceWithLookup
        A new class that adds the lookup table and position selection functionality to cls.
    """

    # Gather motor components, column names, and column suffixes from cls
    motor_components = OrderedDict()
    col_names = []
    col_suffixes = []
    for key in base.__dict__['component_names']:
        component = base.__dict__[key]
        if issubclass(component.cls, EpicsMotor):
            motor_components[key] = component
            col_names.append(key)
            col_suffixes.append(component.suffix)

    
    pos_lookup = OrderedDict(pos_lookup = get_lookup(lut_suffix=lut_suffix, num_rows=num_rows, col_suffixes=col_suffixes, col_names = col_names))
    pos_sel = OrderedDict(pos_sel = Cpt(EpicsSignal, "-" + lut_suffix + "}Pos-Sel", kind = 'hinted', string=True))

    def __init__(self, *args, **kwargs):
        super(type(self), self).__init__(*args, **kwargs)
        self.precision = precision


    def _get_motors(self):
        """
        Return a dictionary of motor components with their current values and setpoints.
        """
        motors = {}
        for key in motor_components:
            motor = {}
            motor["value"] = getattr(self, key).user_readback.get()
            motor["setpoint"] = getattr(self, key).user_setpoint.get()
            motors[key] = motor
        return motors
    
    def _get_table(self):
        """ 
        Return a dictionary representing the lookup table where keys are the row names, 
        and values are dictionaries of column names and their values.
        """
        table = {}
        for i in range(1, num_rows + 1):
            key = f"row{i}"
            row = getattr(self.pos_lookup, key).get_row()
            row_key = next(iter(row))
            table[row_key] = row[row_key]
        return table
    
    def get_all_positions(self):
        """
        Print all possible positions from the lookup table.
        """
        lookup = self._get_table()
        length = len(lookup)
        print(f"\n  {length} Possible Positions:")
        print("----------------------------------")
        for pos_name in lookup:
            if lookup[pos_name] != "Undefined":
                print(f'    {pos_name:_<15} : {tuple(lookup[pos_name].values())}')

    def lookup(self, name : str):
        """
        Look up a position by its name in the lookup table.

        Parameters
        ----------
        name : str
            The name of the position to look up.

        Returns
        -------
        dict
            A dictionary containing the position values for the specified name.

        Raises
        ------
        ValueError
            If the name is not found in the lookup table.
        """
        lookup = self._get_table()

        new_name = (name)

        for pos_name in lookup:
            if pos_name == new_name:
                return lookup[pos_name]
        raise ValueError (f"Could not find {name} in lookup table")
    
    def lookup_by_values(self, pos : tuple[float]):
        """
        Look up a position by its values in the lookup table.

        Parameters
        ----------
        pos : tuple[float]
            A tuple of position values to look up.

        Returns
        -------
        str
            The name of the position that matches the provided values.

        Raises
        ------
        ValueError
            If no matching position is found in the lookup table.
        """
        lookup = self._get_table()
        matched_entry = None
        for lookup_index in range(len(lookup)):
            
            lookup_entry_key = list(lookup.keys())[lookup_index]
            lookup_entry_values = list(lookup[lookup_entry_key].values())
            match = all(round(pos[axis_index], self.precision) == round(lookup_entry_values[axis_index], self.precision) for axis_index in range(len(pos)))
            
                    
            if match == True:
                matched_entry = lookup_entry_key
                break

        if matched_entry == None:
            raise ValueError (f"Could not find {pos} in lookup")
        else:
            return matched_entry    
    
    def where_am_i(self):
        """
        Check consistency between motor values, lookup table, and Pos-Sel value.
        Prints the current motor values, Pos-Sel value, and if they match a preset position.
        """
        motors = self._get_motors()
        pos_sel_val = self.pos_sel.get()

        for axis in motors:
            print("Motor " + axis + " value: " + str(round(motors[axis]["value"], self.precision)))
        print("\nPos-Sel value: ", pos_sel_val)


        motor_values = tuple([motors[motor]['value'] for motor in motors.keys()])
        try:
            matched_entry = self.lookup_by_values(motor_values)
            
        except ValueError:
            matched_entry = None

        
        if (matched_entry is None) or (pos_sel_val != matched_entry):
            print("\nYour motor values and/or Pos-Sel do not match one of the preset positions")
            print("Use " + self.name + ".get_all_positions() to see all preset positions.")
        else:
            print("\nYour motor and pos-sel values matched the position: " + matched_entry + " = ", self.lookup(matched_entry))

    def set_pos_sel(self, pos: str | tuple) -> Status:
        """
        Set the Pos-Sel value to a specific position or by its name.

        Parameters
        ----------
        pos : str | tuple
            The position to set, either as a name (str) or as a tuple of values (tuple).

        Returns
        -------
        Status
            A Status object.
        """

        if isinstance(pos, tuple):
            pos = self.lookup_by_values(pos)
        
        return self.pos_sel.set(str(pos))
    
    def _sync_pos_sel(self, status = None):
        """
        Synchronize the Pos-Sel value with the current motor setpoints.
        Parameters
        ----------
        status : optional
            The status of the move operation, if applicable.
        """
        self.set_pos_sel("Undefined")
        motors = self._get_motors()
        motor_values = tuple([motors[axis]["setpoint"] for axis in motors])
        self.set_pos_sel(motor_values)

    def set(self, size : str | tuple) -> Status:
        """
        Set the motors to a specific position or by its position name and sync it with the pos_sel signal.


        Parameters
        ----------
        size : str | tuple
            The position to set, either as a name (str) or as a tuple of values (tuple).

        Returns
        -------
        Status
            A Status object indicating the status of the move operation.
        """

        if isinstance(size, str):
            values = list(self.lookup(size).values())
        else: values = list(size)
        

        motors = self._get_motors()
        axes = list(motors.keys())

        self.set_pos_sel("Undefined")
        init_motor = getattr(self, axes[0])
        move_status = init_motor.set(values[0])

        for axis_index in range(1, len(axes)):
            curr_motor = getattr(self, axes[axis_index])
            move_status = move_status & curr_motor.set(values[axis_index])

        move_status.add_callback(self._sync_pos_sel)
        return move_status

    # Create the class dictionary with all methods and attributes
    clsdict = OrderedDict(
            __doc__=(f"DeviceWithLookup Device"),
            _default_read_attrs=["pos_lookup", "pos_sel"].extend(col_names), # Assign read-attrs to new class
            _default_configuration_attrs=[],
            # Add new methods
            __init__ = __init__,
            _get_motors = _get_motors,
            _get_table = _get_table,
            get_all_positions = get_all_positions,
            lookup = lookup,
            lookup_by_values = lookup_by_values,
            where_am_i = where_am_i,
            set_pos_sel = set_pos_sel,
            _sync_pos_sel = _sync_pos_sel,
            set = set
        )

    # Add new components
    clsdict = clsdict | motor_components | pos_sel | pos_lookup
    
    # Create the new class with the new  components and methods
    DeviceWithLookup = type("DeviceWithLookup", (base,), clsdict, **{})

    return DeviceWithLookup