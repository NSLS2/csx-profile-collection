from bluesky.plan_stubs import null, mv, mvr
from bluesky.plans import count

import os, subprocess, inspect
from rich import print as cprint
from typing import Callable
from ..startup.optics import ps_front_end, fs_diag1_x, m1a, fe_slt
from ..startup.accelerator import epu1, epu2, canter
from ..startup.detectors import cam_fs

## The devices used below need to be replaced with the imported devices

# sd.baseline.extend([FEslt.x.gap.readback, 
#           FEslt.x.cent.readback, 
#           FEslt.y.gap.readback, 
#           FEslt.y.cent.readback,
#           fs_diag1_x.pos_sel,
#           fs_diag1_x.x.user_readback,
#           canter,
#           phaser.user_readback])


def colored(text, tint='white', attrs=[], end=None):
    '''
    A simple wrapper around rich.print
    '''
    if not False:
        tint = tint.lower()
        this = f'[{tint}]{text}[/{tint}]'
        if end is not None:
            cprint(f'[{tint}]{text}[/{tint}]', end=end)
        else:
            cprint(f'[{tint}]{text}[/{tint}]')
    else:
        print(text)


def whisper(text, end=None):
    '''Light gray text'''
    colored(text, 'bold black', end=end)

def print_dict(dict):
    for key in dict:
        print(f"\t{key}: {dict[key]}")

# A mapping of canter states to coordinates for FE_shutter and m1a starting positions
canter_map = {"canted" : {"FEslt" : {"x.gap" : 7.000, "y.gap" : 1.800, "x.cent" : 0, "y.cent" : 0.650}, "m1a" : {"x" : 0, "y" : -2.410, "z" : -27.620, "pit" : 6.175, "yaw" : 0, "rol" : 2.400}}, 
                "straight" : {"FEslt" : {"x.gap" : 4.000, "y.gap" : 1.800, "x.cent" : 0, "y.cent" : 0.650}, "m1a" : {"x" : 0, "y" : 0.800, "z" : 27.500, "pit" : 6.525, "yaw" : 0, "rol" : 4.400}}}

def make_fluo_img(md):
    
    yield from count([cam_fs1_hdf5], num=4, md={'purpose':'source check', 'source check':md})

class SourceCheck():
    
    

    def source_check_manual(self):
        '''
        Print the menu for the source check and prompt the user for a choice.
        '''

        actions = {'0': ('Prep',   'preparation'),
                   '1': ('Step1',   'detuned source'),
                   '2': ('Step2',   'ios source'),
                   '3': ('Step3',   'csx source'),
                   '4': ('Step4',   'source'),
                   '5': ('Step5',   'check slits1'),
                   '6': ('Step6',   'check m1a pos'),
                   '7': ('Step7',   'check pink beam'),
                   '8': ('Step8',   'check pink beam & slits'),
                   '9': ('Return',     'return to ops'),
                   '10': ('Quit',   'Quit')
        }
        
        print('\n  CHOICES\n======================================''')

        for i in range(0,11):
            text  = actions[str(i)][1]
            try:
                other = actions[chr(64+i)][1]
                print(f' {i}. {text:37} ')
            except:
                print(f' {i}. {text:37}')
        print('')
        choice = input(" What do you want to do? ")
        choice = choice.upper()
        print('\n')
        def bailout():
            whisper('doing nothing')
            yield from null()
        thing = 'do_nothing'
        if choice in actions:
            thing = f'do_{actions[choice][0]}'
        return getattr(self, thing, lambda: bailout)()


    def end_step(self, next_step, next_step_name):
        """Handle the end of a source check step.
        
        Prompt the user to
                
                (1) Continue to next step
                (2) Go back to the source check menu
                (3) Quit the source check
                
        and proceed with their choice.
        """

        print("End of step - Choose how to proceed:\n")
        print(f" 1. Continue to next step: {next_step_name}")
        print(" 2. Quit and go to menu")
        print(" 3. Quit source check (default)")
        i = input()
        match i:
            case "1":
                next_step()
            case "2":
                self.source_check_manual()
            case _:
                return 

         
    def pause(self, prompt, action : tuple | Callable, current_step : Callable, current_prompt : Callable):
        """Handle an interruption in the source check process.
        
        Prompt the user to 
        
                (1) quit
                (2) continue the step
                (3) restart the current step
                (4) go back to the source check menu

        and return True to continue the step, otherwise, perform the desired action.

        Parameters
        ----------
        current_step : function
                The function representing the current step in the source check process.
        """
        print("\nSource check paused. What would you like to do?")
        print("1. Continue from last prompt" \
                "\n2. Return to source check menu" \
                "\n3. Restart step" \
                "\n4. Quit source check (default)")
        match input():
                case "1": 
                        return current_prompt(prompt, action, current_step)

                case "2":
                        self.source_check_manual()
                        return False

                case "3":
                        current_step()
                        return False
                
                case _:
                        return False
            

    # THIS FUNCTION IS CURRENTLY SET TO PRINT OUT THE YIELDS RATHER THAN YIELD THEM
    def confirm_default_n(self, prompt, action : tuple | Callable, current_step : Callable):
        """Prompt the user to confirm an action with a default response of 'n'.
            Performs action and returns True if the user confirms otherwise returns 
            result of the pause() function.

        Parameters
        ----------
        prompt : str
                The message to display to the user.
        action : function
                The plan to execute if the user confirms.
        current_step : function
                The function representing the current step in the source check process.
        """
        match input(prompt + "  "):
                case "y":
                        if isinstance(action, Callable):
                            #  yield from action()
                             print(f"yield from {action.__name__}")
                        if isinstance(action, tuple):
                            plan, motor, target = action
                            # yield from plan(signal, target)
                            print(f"yield from {plan.__name__}({motor}, {target})")
                        return True

                case _ : return self.pause(prompt, action, current_step, self.confirm_default_n)
        
    
    def confirm_default_y(self, prompt, action: tuple | Callable, current_step : Callable):
        """Prompt the user to confirm an action with a default response of 'y'.
            Returns returns result of the pause() function if user refuses, otherwise 
            performs action and returns True.

        Parameters
        ----------
        prompt : str
                The message to display to the user.
        action : function
                The plan to execute if the user confirms.
        current_step : function
                The function representing the current step in the source check process.
        """
        match input(prompt + "  "):
                case "n" : return self.pause(prompt, action, current_step, self.confirm_default_y)
             
                case _ :
                        if isinstance(action, Callable):
                            #  yield from action()
                             print(f"yield from {action.__name__}")
                        if isinstance(action, tuple):
                            plan, motor, target = action
                            # yield from plan(signal, target)
                            print(f"yield from {plan.__name__}({motor}, {target})")
                        return True

    def prompt_and_act(self, prompts, actions, defaults, current_step):
        """Prompt the user with a list of prompts and actions, and execute the actions based on the user's input.
        
        Parameters
        ----------
        prompts : list of str
                The messages to display to the user.
        actions : list of tuple | Callable
                The plans to execute if the user confirms.
        defaults : list of str
                The default responses for each prompt.
        current_step : Callable
                The function representing the current step in the source check process.
        """
        for prompt, action, default in zip(prompts, actions, defaults):
            if default == 'y':
                if not self.confirm_default_y(prompt, action, current_step):
                    return False
            else:
                if not self.confirm_default_n(prompt, action, current_step):
                    return False
        return True



    ### --------------- STEPS START HERE --------------- ###


    def do_Prep(self):

        print("\nSource check preparation")
        print("--------------------------")

        # Check canter position
        if (1 > canter.get()) and (canter.get() > -1): # 'canted' position should be 0 but readback value deviates
            canting_pos = "canted" 
        else:
            canting_pos = "straight"
        
        print("\n\tCheck Canting Position")
        print("\t------------------------")
        if not self.confirm_default_y(f"\n\tThe current canter position is <{canting_pos}>. Proceed ([y]/n)?", lambda:None, self.do_Prep): return


        # Record starting positions using setpoints
        print("\n\tRecord OPS")
        print("\t-----------")
        m1a_ops = {"x" : round(m1a.x.setpoint.get(), 4), 
                   "y" : round(m1a.y.setpoint.get(), 4), 
                   "z" :  round(m1a.z.setpoint.get(), 4), 
                   "pit" : round(m1a.pit.setpoint.get(), 4), 
                   "yaw" : round(m1a.yaw.setpoint.get(), 4), 
                   "roll" : round(m1a.rol.setpoint.get(), 4)}
        
        FEslt_ops = {"x_gap" : round(FEslt.x.gap.setpoint.get(), 4), 
                     "y_gap" : round(FEslt.y.gap.setpoint.get(), 4), 
                     "x_cent" : round(FEslt.x.cent.setpoint.get(), 4), 
                     "y_cent" : round(FEslt.y.cent.setpoint.get(), 4)}
        
        epu1_ops = {"gap" : round(epu1.gap.readback.get(), 4), "phase" : round(epu1.phase.readback.get(), 4)}
        epu2_ops = {"gap" : round(epu2.gap.readback.get(), 4), "phase" : round(epu2.phase.readback.get(), 4)}

        RE.md['source_check_noc'] = {"m1a" : m1a_ops, "FEslt" : FEslt_ops, "epu1" : epu1_ops, "epu2" : epu2_ops, "canter" : canting_pos}
        
        print(f"\n\tRecording current M1a position as \n")
        print_dict(m1a_ops)
        print(f"\n\tRecording current FEslt position as operating position as \n")
        print_dict(FEslt_ops)
        print(f"\n\tRecording current EPU1 position as \n")
        print_dict(epu1_ops)
        print(f"\n\tRecording current EPU1 position as \n")
        print_dict(epu2_ops)

        

        # Make sure FE shutter is closed
        print("\n\tFE Shutter")
        print("\t-----------")
        if FE_shutter.status.get() != 'Closed':
            if not self.confirm_default_y("\n\tClose FE Shutter? ([y]/n)", (mv, FE_shutter, 'Cls'), self.do_Prep): return
            
        else:
            print("\n\tFE Shutter is closed")

        # Make sure EPU phases are 0
        print("\n\tEPU Phases")
        print("\t-----------")

        if (epu1.phase.setpoint.get() != 0):

            if not self.confirm_default_n((f"\n\tThe current EPU1 phase is {epu1_ops['phase']}. Set EPU1 phase to 0? (y/n)"), 
                                        (mv, epu1.phase, 0), self.do_Prep): return
            
        else:
            print("\n\tEPU1 phase is 0")
    
        if (epu2.phase.setpoint.get() != 0):

            if not self.confirm_default_n((f"\n\tThe current EPU2 phase is {epu2_ops['phase']}. Set EPU2 phase to 0? (y/n)"), 
                                        (mv, epu2.phase, 0), self.do_Prep): return
            
        else:
            print("\n\tEPU2 phase is 0")

        print()
        self.end_step(self.do_Step1, "Detuned Source")


    def do_Step1(self):

        print("\nStep 1 - detuned source\n")
        print("--------------------------")

        prompts = [
            "\n\tSet EPU1 Gap to 100. Confirm ([y]/n).",
            "\n\tSet EPU2 Gap to 100. Confirm ([y]/n).",
            "\n\tOpen FE slits. Confirm (y/[n]).",
            "\n\tMove m1a to 'out' position. Confirm (y/[n]).",
            "\n\tOpen FE shutter. Confirm (y/[n]).",
            "\n\tTake photo of FSdiag. Confirm ([y]/n)."
        ]

        actions = [
            (mv, epu1.gap, 100),
            (mv, epu2.gap, 100),
            (mvr, FEslt.y.gap, -3),
            (mvr, m1a.y, -6),
            (mv, FE_shutter, 'Opn'),
            make_fluo_img('BM')
        ]
        
        defaults = ['y', 'y', 'n', 'n', 'n', 'y']

        if not self.prompt_and_act(prompts, actions, defaults, self.do_Step1): return

        print()
        self.end_step(self.do_Step2, "IOS Source")


    def do_Step2(self):

        print("\nStep 2 - IOS source\n")
        print("--------------------------")

        prompts = [
            "\n\tSet EPU1 Gap to 82. Confirm ([y]/n)", 
            "\n\tTake photo of FSdiag. Confirm ([y]/n)."
        ]

        actions = [
            (mv, epu1.gap, 82),
            make_fluo_img('EPU:2')
        ]

        defaults = ['y', 'y']

        if not self.prompt_and_act(prompts, actions, defaults, self.do_Step2): return

        print()
        self.end_step(self.do_Step3, "CSX Source")


    def do_Step3(self):

        print("\nStep 3 - csx source\n")
        print("--------------------------")

        prompts = [
            "\n\tSet EPU1 Gap to 100. Confirm ([y]/n)", 
            "\n\tSet EPU2 Gap to 85. Confirm ([y]/n)", 
            "\n\tTake photo of FSdiag. Confirm ([y]/n)."
        ]

        actions = [
            (mv, epu1.gap, 100),
            (mv, epu2.gap, 85),
            make_fluo_img('EPU:1')
        ]

        defaults = ['y', 'y', 'y']

        if not self.prompt_and_act(prompts, actions, defaults, self.do_Step3): return
        print()
        self.end_step(self.do_Step4, "Source")


    def do_Step4(self):

        print("\nStep 4 - source\n")
        print("--------------------------")

        prompts = [
            "\n\tSet EPU1 Gap to 82. Confirm ([y]/n)", 
            "\n\tTake photo of FSdiag. Confirm ([y]/n)."
        ]

        actions = [
            (mv, epu1.gap, 82),
            make_fluo_img('BOTH')
        ]

        defaults = ['y', 'y']

        if not self.prompt_and_act(prompts, actions, defaults, self.do_Step4): return

        print()
        self.end_step(self.do_Step5, "Check Slits")

    def do_Step5(self):
        
        print("\nStep 5 - check slits")
        print("--------------------------")

        prompts = [
             f"\n\tSet X Gap of FE slits to operating position: {self.FEslt_ops['x_gap']}. Confirm (y/[n])",
            "\n\tTake photo of FSdiag. Confirm ([y]/n)"
        ]

        actions = [
            (mv, FEslt.x.gap, RE.md["source check"]["FEslt"]['x_gap']),
            make_fluo_img('BOTH FEslt')
        ]

        defaults = ['n', 'y']

        if not self.prompt_and_act(prompts, actions, defaults, self.do_Step5): return

        print()
        self.end_step(self.do_Step6, "Check M1a Position")

    def do_Step6(self):

        print("\nStep 6 - check m1a pos\n")
        print("--------------------------")


        prompts = [
            "\n\tClose FE shutter. Confirm ([y]/n)",
            f"\n\tMove m1a to operating position: {self.m1a_ops}. Confirm (y/[n])",
            "\n\tOpen FE slits. Confirm (y/[n])",
            "\n\tTake photo of FSdiag. Confirm ([y]/n)"
        ]

        actions = [
            (mv, FE_shutter, 'Close'),
            (mv, m1a, RE.md["source_check"]["m1a"]),
            FEslt.mv_open,
            make_fluo_img('EPU:1 M1A FEslt')
        ]

        defaults = ['y', 'n', 'n', 'y']

        if not self.prompt_and_act(prompts, actions, defaults, self.do_Step6): return

        print()
        self.end_step(self.do_Step7, "Check Pink Beam")

    def do_Step7(self):

        print("\nStep 7 - check pink beam\n")
        print("--------------------------")


        prompts = [
            "\n\tClose FE shutter. Confirm ([y]/n)",
            "\n\tSet FS diag to Pink Beam position. Confirm (y/[n])",
            "\n\tOpen FE shutter. Confirm (y/[n])",
            "\n\tTake photo of FSdiag. Confirm ([y]/n)"
        ]

        actions = [
            (mv, FE_shutter, 'Close'),
            (mv, fs_diag1_x, 'Pink Beam'),
            (mv, FE_shutter, 'Open'),
            make_fluo_img('PINK')
        ]

        defaults = ['y', 'n', 'n', 'y']

        if not self.prompt_and_act(prompts, actions, defaults, self.do_Step7): return
        

        print()
        self.end_step(self.do_Step8, "Check Pink Beam & Slits")

    def do_Step8(self):

        print("\nStep 8 - check pink beam & slits\n")
        print("-----------------------------------")

        prompts = [
            "\n\tClose FE shutter. Confirm ([y]/n)",
            f"\n\tSet FEslt to operating position: {self.FEslt_ops}. Confirm (y/[n])",
            "\n\tTake photo of FSdiag. Confirm ([y]/n)"
        ]

        actions = [
            (mv, FE_shutter, 'Cls'),
            (mv, FEslt, RE.md['source check'].FEslt_ops),
            make_fluo_img('PINK FEslit')
        ]

        defaults = ['y', 'n', 'y']

        if not self.prompt_and_act(prompts, actions, defaults, self.do_Step8): return

        print()
        self.end_step(self.do_Step9, "Return to OPS")


    def do_ReturnToOPS(self):

        print("\nEnd - return to operating positions\n")
        print("-------------------------------------")

        prompts = [
            "\n\tClose FE shutter. Confirm ([y]/n)",
            "\n\tSet FS diag to 'out'. Confirm ([y]/n)",
            "\n\tOpen FE shutter. Confirm (y/[n])"
        ]

        actions = [
            (mv, FE_shutter, 'Cls'),
            (mv, fs_diag1_x, 'out'),
            (mv, FE_shutter, 'Opn')
        ]

        defaults = ['y', 'y', 'n']

        if not self.prompt_and_act(prompts, actions, defaults, self.do_ReturnToOPS): return
        RE.md.pop("source_check")




    def do_Quit(self):
        return
    

prompt = SourceCheck()





