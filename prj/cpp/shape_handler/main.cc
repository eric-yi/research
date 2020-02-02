#include <stdlib.h>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "shapefil.h"

using namespace std;

#define DATA_PATH "data"

string GetFilePath(const char *filename, const char *ext)
{
    string path(DATA_PATH);
    path += "/";
    path += filename;
    path += ".";
    path += ext;
    return path;
}

void Handle(const char *filename)
{
    string dbf_path = GetFilePath(filename, "dbf");
    DBFHandle dbf_handle = DBFOpen(dbf_path.c_str(), "rb");
    if (dbf_handle == nullptr)
    {
        cout << "Error: open dbf file error: " << dbf_path.c_str() << endl;
        exit(2);
    }
    int record_count = DBFGetRecordCount(dbf_handle);
    cout << "record count: " << record_count << endl;

    string shp_path = GetFilePath(filename, "shp");
    SHPHandle shp_handle = SHPOpen(shp_path.c_str(), "rb");
    if (shp_handle == nullptr)
    {
        cout << "Error: open shape file error: " << shp_path.c_str() << endl;
        exit(1);
    }
    int entities;
    int shape_type;
    double min_bound;
    double max_bound;
    SHPGetInfo(shp_handle, &entities, &shape_type, &min_bound, &max_bound);
    cout << "shape info: {entities: " << entities
         << ", shape type:" << shape_type
         << ", min bound:" << min_bound
         << ", max bound:" << max_bound
         << "}" << endl;

    shp_path = GetFilePath(filename, "shp");
    shp_handle = SHPOpen(shp_path.c_str(), "rb");
    if (shp_handle == nullptr)
    {
        cout << "Error: open shape file error: " << shp_path.c_str() << endl;
        exit(1);
    }

    double width = 1200.0;
    double height = 600.0;
    cv::Mat image(height, width, CV_8U, 100);
    double x_det = width / (14.015 - min_bound);
    double y_det = height / (51.2 - max_bound);
    for (int i = 0; i < entities; i++)
    {
        SHPObject *shp_object = SHPReadObject(shp_handle, i);
        int o_type = shp_object->nSHPType;
        int o_id = shp_object->nShapeId;
        int parts = shp_object->nParts;
        int *part_start = shp_object->panPartStart;
        int *part_type = shp_object->panPartType;
        int vertices = shp_object->nVertices;
        double *pad_x = shp_object->padfX;
        double *pad_y = shp_object->padfY;
        double *pad_z = shp_object->padfZ;
        double *pad_m = shp_object->padfM;
        double x_min = shp_object->dfXMin;
        double y_min = shp_object->dfYMin;
        double z_min = shp_object->dfZMin;
        double m_min = shp_object->dfMMin;
        double x_max = shp_object->dfXMax;
        double y_max = shp_object->dfYMax;
        double z_max = shp_object->dfZMax;
        double m_max = shp_object->dfMMax;
        int is_used = shp_object->bMeasureIsUsed;
        int read_object = shp_object->bFastModeReadObject;

        int v = 0;
        double px;
        double py;
        double pz;
        double pm;
        double p1_x;
        double p1_y;
        double p2_x;
        double p2_y;
        while (v < vertices)
        {
            if (v > 0)
            {
                p1_x = (px - min_bound) * x_det;
                p1_y = height - (py - max_bound) * y_det;
                p2_x = (*(pad_x + v) - min_bound) * x_det;
                p2_y = height - (*(pad_y + v) - max_bound) * y_det;
                cv::line(image,
                         cv::Point(p1_x, p1_y),
                         cv::Point(p2_x, p2_y),
                         cv::Scalar(0, 0, 255), 1);
            }
            px = *(pad_x + v);
            py = *(pad_y + v);
            pz = *(pad_z + v);
            pm = *(pad_m + v);
            // cout << "pad: {x:" << px << ", y:" << py << ", z:" << pz << ", m:" << pm << "}" << endl;
            // cout << "line: (" << p1_x << "," << p1_y << ") - (" << p2_x << "," << p2_y << ")" << endl;
            // cout << "line: (" << p1_x << endl;
            v++;
        }


        // cout << "shape " << i << " {id:" << o_id
        //      << ", type:" << o_type
        //      << ", parts:" << parts
        //      << ", part start:" << *part_start
        //      << ", part type:" << *part_type
        //      << ", vertices:" << vertices
        //      << ", pad: {x:" << *pad_x << ", y:" << *pad_y << ", z:" << *pad_z << ", m:" << *pad_m << "}"
        //      << ", min: {x:" << x_min << ", y:" << y_min << ", z:" << z_min << ", m:" << m_min << "}"
        //      << ", max: {x:" << x_max << ", y:" << y_max << ", z:" << z_max << ", m:" << m_max << "}"
        //      << ", measure is used:" << is_used
        //      << ", fast mode read object:" << read_object
        //      << "}" << endl;
    }
    cv::imshow("Image", image);
    cv::waitKey(0);
}

int main(int argc, char *argv[])
{
    cout << "==== Shape Handler ==== " << endl;
    Handle("gis_osm_roads_07_1");
    return 0;
}