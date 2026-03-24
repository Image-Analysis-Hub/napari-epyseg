import numpy as np

from napari.utils.history import get_save_history, update_save_history 
from napari import current_viewer
from magicgui import magicgui
from napari.layers import Image
from napari.utils import notifications as nt
from napari.utils import progress # type: ignore

import pathlib
import os
from napari_epyseg.call_epyseg import run_epyseg, writeTif

def start_epyseg():
    """ Import only if it will use napari """
    global cdir
    hist = get_save_history()
    cdir = hist[0]
    viewer = current_viewer()
    paras = dict()
    paras["overlap_width"] = 32
    paras["overlap_height"] = 32
    paras["tile_width"] = 256
    paras["tile_height"] = 256
    paras["norm_min"] = 0
    paras["norm_max"] = 1
    return choose_parameters( viewer, paras )

def choose_parameters( viewer, parameters ):
    @magicgui(call_button="Save segmentation",
            save_file={"label": "Segmentation filename", "mode": "w"},
            )
    def save_interface(
        save_file = pathlib.Path(os.path.join(cdir)),
        ):
        """ Save file interface """
        if not str(save_file).endswith(".tif"):
            nt.show_warning("Unvalid segmentation filename (should be a tif file), set it to a correct file path")
            return
        update_save_history(save_file)
        save_segmentation_file( str(save_file), viewer )

    def show_model_file():
        """ Show/hide the model file interface (if custom is selected) """
        get_parameters.model_file.visible = (get_parameters.model.value == "custom model")
    
    def show_parameters():
        """ Handle advanced parameters visibility """
        get_parameters.overlap_width.visible = (get_parameters.advanced.value == True)
        get_parameters.overlap_height.visible = (get_parameters.advanced.value == True)
        get_parameters.tile_width.visible = (get_parameters.advanced.value == True)
        get_parameters.tile_height.visible = (get_parameters.advanced.value == True)
        get_parameters.normalization_min_percentile.visible = (get_parameters.advanced.value == True)
        get_parameters.normalization_max_percentile.visible = (get_parameters.advanced.value == True)

    def show_channel():
        """ Show/hide channel choice depending on selected layer """
        shape = get_parameters.image.value.data.shape
        print(shape)
        get_parameters.channel.visible = (len(shape)>3)

    def display_channel():
        """ Display the selected channel """
        img = viewer.layers[ get_parameters.image.value.name ].data
        chan_axis = 1
        ## check that the color channel axis is the second one
        if img.shape[0] < img.shape[1]:
            chan_axis = 0
        chan_nb = get_parameters.channel.value
        if (chan_nb < 0) or (chan_nb>img.shape[chan_axis]):
            nt.show_warning("Invalid channel number, set a value in the correct range: [0-"+str(img.shape[chan_axis]-1)+"]")
            return 
        viewer.dims.set_point( chan_axis, get_parameters.channel.value )
        


    @magicgui(call_button="Segment",
            image={'label': 'Pick an Image'},
            model={'label': 'Model to use', "choices": ['epyseg default(v2)', 'custom model']},
            model_file = {'label': 'Custom model file (.h5)'},
            normalization_min_percentile={"widget_type": "LiteralEvalLineEdit"},
            normalization_max_percentile={"widget_type": "LiteralEvalLineEdit"},
            tile_width={"widget_type": "LiteralEvalLineEdit"},
            tile_height={"widget_type": "LiteralEvalLineEdit"},
            overlap_width={"widget_type": "LiteralEvalLineEdit"},
            overlap_height={"widget_type": "LiteralEvalLineEdit"},
            )
    def get_parameters( 
            image: Image,
            channel: int=0,
            model = "epyseg default(v2)",
            model_file = pathlib.Path(cdir),
            advanced = False,
            normalization_min_percentile = parameters["norm_min"],
            normalization_max_percentile = parameters["norm_max"],
            tile_width = parameters["tile_width"],
            tile_height = parameters["tile_height"],
            overlap_width = parameters["overlap_width"],
            overlap_height = parameters["overlap_height"],
            ):
        """ Choose the parameters to run Epyseg on selected file """
        parameters["tile_width"] = tile_width
        parameters["tile_height"] = tile_height
        parameters["overlap_width"] = overlap_width
        parameters["overlap_height"] = overlap_height
        parameters["norm_min"] = normalization_min_percentile
        parameters["norm_max"] = normalization_max_percentile
        parameters["model"] = model
        parameters["model_file"] = str(model_file)
        img = image.data
        chan_axis = 1
        if len(img.shape) > 3:
            if img.shape[0] < img.shape[1]:
                img = img[channel,]
                chan_axis = 0
            else:
                img = img[:,channel,]
        viewer.window._status_bar._toggle_activity_dock( True )
        progress_bar = progress( len(img) )
        progress_bar.set_description( "Running epyseg on all frames..." )
        progress_bar.update(0)
        res = run_epyseg( img, parameters, progress_bar=progress_bar )
        viewer.window._status_bar._toggle_activity_dock( False )
        if len(res.shape) < len(image.data.shape):
            res = np.expand_dims( res, axis=chan_axis )
        viewer.add_image( res, scale=image.scale, blending="additive", name="Segmentation" )
        viewer.window.add_dock_widget( save_interface )
    
    get_parameters.model.changed.connect( show_model_file )
    get_parameters.image.changed.connect( show_channel )
    get_parameters.channel.changed.connect( display_channel )
    get_parameters.model_file.visible = False
    get_parameters.advanced.changed.connect( show_parameters )
    wid = viewer.window.add_dock_widget( get_parameters )
    show_parameters()
    return wid

def save_segmentation_file( filename, viewer ):
    """ Save the segmentation results to file """
    if "Segmentation" not in viewer.layers:
        nt.show_warning("No segmentation found")
        return
    lay = viewer.layers["Segmentation"]
    laydata = lay.data
    if len(laydata.shape) > 3:
        laydata = lay.data[:,0,:,:]

    writeTif( laydata, filename, lay.scale, "uint8", what="Segmentation", nt=nt )
