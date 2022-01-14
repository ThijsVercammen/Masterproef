package org.pytorch.testapp;

import android.graphics.RectF;

public class Recognition {

    private final String id;
    private final String title;
    private final Float confidence;
    private RectF location;
    private int color = 0;

    public Recognition(
            final String id, final String title, final Float confidence, final RectF location, final int color) {
        this.id = id;
        this.title = title;
        this.confidence = confidence;
        this.location = location;
        this.color =color;
    }

    public String getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public Float getConfidence() {
        return confidence;
    }

    public RectF getLocation() {
        return new RectF(location);
    }

    public void setLocation(RectF location) {
        this.location = location;
    }

    public int getColor() {return color;}

    public void setColor(int drawn) {this.color = color;}
}
