from bluesky.callbacks.best_effort import BestEffortCallback


class BECwithTicks(BestEffortCallback):
    def _set_up_plots(self, doc, stream_name, columns):
        super()._set_up_plots(doc, stream_name, columns)
        # Now loop through all of the set up plots and update tick params.
        # This will keep the axes aligned, but remove the assumption that you only want partial labels.

        for scatter in self._live_scatters.get(doc["uid"], {}).values():
            scatter.ax.tick_params(labelbottom=True, labelleft=True)
        for grid in self._live_grids.get(doc["uid"], {}).values():
            grid.ax.tick_params(labelbottom=True, labelleft=True)
